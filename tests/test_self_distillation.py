import json
from pathlib import Path
import queue
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from carla_delay_common import CameraRig, read_routes
from delay_dataset import DelayedJSONActionDataset, discover_sequences, split_sequences, stale_index
from finetune_time_adapter_from_json import active_logits, distillation_loss, run_epoch
from pcla_agents.wor.rails.camera_io import camera_specs, crop_rgb, stitch_rgb
from pcla_agents.wor.rails.models.main_model import CameraModel
from pcla_agents.wor.rails.models.time_adapter import DelayFeatureAdapter, load_delay_adapter


def config(two_cam=False, all_speeds=True):
    return dict(num_steers=3, num_throts=2, num_speeds=4, seg_channels=[1],
                all_speeds=all_speeds, use_narr_cam=two_cam, imagenet_pretrained=False,
                min_speeds=0., max_speeds=8., wide_crop_top=0, narr_crop_bottom=0,
                camera_x=1.5, camera_z=2.4)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def make_sequence(root, name, route='r1', times=(0, .05, .1, .15, .2, .25), town='Town01'):
    directory = root / name
    directory.mkdir()
    frames = []
    for i, timestamp in enumerate(times):
        wide, narrow = f'w{i}.png', f'n{i}.png'
        Image.fromarray(np.full((32, 48, 3), i * 20, dtype=np.uint8)).save(directory / wide)
        Image.fromarray(np.full((32, 32, 3), i * 20 + 1, dtype=np.uint8)).save(directory / narrow)
        frames.append(dict(timestamp=timestamp, wide_rgb=wide, narr_rgb=narrow, spd=i + 1.,
                           cmd=i % 6, yaw_rate=.1 * i, previous_control=[.1, .2, 0]))
    (directory / 'data.json').write_text(json.dumps(dict(town=town, route_id=route, frames=frames)))
    return directory


def test_no_q_labels_and_live_state_timestamp_pairs(tmp_path):
    make_sequence(tmp_path, 'one')
    sequences = discover_sequences(tmp_path)
    ds = DelayedJSONActionDataset(sequences, config())
    assert len(ds) == 10  # two eligible current frames, five delays each
    pair = ds[4]  # t=.2, stale=0, requested=.2
    assert pair['latency_s'] == pytest.approx(.2)
    assert pair['timestamp'] == pytest.approx(.2)
    assert pair['speed'] == 5 and pair['command'] == 4
    assert pair['ego_state'].tolist() == pytest.approx([5, .4, .1, .2, 0])
    assert np.all(pair['stale_wide'] == 0) and np.all(pair['current_wide'] == 80)
    assert pair['stale_narr'].shape == (32, 32, 3)  # crop=0 never means empty
    assert np.array_equal(ds[0]['current_wide'], ds[0]['stale_wide'])


def test_irregular_timestamps_and_gaps(tmp_path):
    assert stale_index([0, .052, .1], 2, .05) == 0  # .052 is too recent
    assert stale_index([0, .05, .1, .15], 3, .1) == 1  # tolerate fp rounding
    make_sequence(tmp_path, 'gap', times=(0, .05, .5, .55))
    with pytest.raises(ValueError, match='No complete delay windows'):
        DelayedJSONActionDataset(discover_sequences(tmp_path), config())


@pytest.mark.parametrize('times', [(0, 0), (0, float('nan')), (.1, 0)])
def test_invalid_timestamps_fail(tmp_path, times):
    make_sequence(tmp_path, 'bad', times=times)
    with pytest.raises(ValueError, match='timestamps'):
        discover_sequences(tmp_path)


def test_split_groups_repeated_routes_and_rejects_leakage(tmp_path):
    make_sequence(tmp_path, 'a', 'shared')
    make_sequence(tmp_path, 'b', 'shared')
    make_sequence(tmp_path, 'c', 'new', town='Town02')
    sequences = discover_sequences(tmp_path)
    train, val, manifest = split_sequences(sequences, heldout_towns=['Town02'])
    assert {s['id'] for s in train} == {'a', 'b'} and [s['id'] for s in val] == ['c']
    assert split_sequences(sequences, manifest=manifest)[2] == manifest
    with pytest.raises(ValueError, match='leakage'):
        split_sequences(sequences, manifest=dict(train=['a'], val=['b', 'c']))
    with pytest.raises(ValueError, match='partition'):
        split_sequences(sequences, manifest=dict(train=['a'], val=['c']))


def test_rgb_stitch_order_and_crop_contract():
    panels = [np.full((240, 160, 4), (i, i + 1, i + 2, 255), dtype=np.uint8) for i in (10, 20, 30)]
    narrow = np.full((240, 384, 4), (40, 41, 42, 255), dtype=np.uint8)
    wide, narrow = stitch_rgb(panels, narrow)
    assert wide.shape == (240, 480, 3)
    for i, v in enumerate((10, 20, 30)):
        assert wide[0, i * 160].tolist() == [v + 2, v + 1, v]
    wide, narrow = crop_rgb(wide, narrow, dict(wide_crop_top=48, narr_crop_bottom=16))
    assert wide.shape == (192, 480, 3) and narrow.shape == (224, 384, 3)
    assert [s['yaw'] for s in camera_specs(config())] == [-55, 0, 55, 0]


def test_kl_direction_and_stop_gradient():
    student = torch.tensor([[1., -1.]], requires_grad=True)
    teacher = torch.tensor([[-2., 2.]], requires_grad=True)
    features = torch.ones(1, 3, requires_grad=True)
    target = torch.zeros(1, 3, requires_grad=True)
    loss, kl, mse = distillation_loss(student, teacher, features, target, .25)
    p, q = student.softmax(-1), teacher.softmax(-1)
    expected = (p * (p.log() - q.log())).sum()
    assert torch.allclose(kl, expected)
    assert torch.allclose(loss, expected + .25)
    loss.backward()
    assert student.grad is not None and features.grad is not None
    assert teacher.grad is None and target.grad is None
    _, forward_kl, _ = distillation_loss(student, teacher, features, target, direction='teacher_student')
    expected_forward = (q * (q.log() - p.log())).sum()
    assert torch.allclose(forward_kl, expected_forward)


@pytest.mark.parametrize('ego_dim', [0, 1, 5])
def test_adapter_identity_after_training_and_checkpoint_roundtrip(ego_dim):
    adapter = DelayFeatureAdapter(8, hidden_dim=16, max_latency_s=.2, ego_dim=ego_dim)
    features = torch.randn(3, 8)
    ego = torch.randn(3, ego_dim)
    assert torch.equal(adapter(features, torch.tensor([0, .1, .2]), ego), features)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        adapter(features, torch.full((3,), .2), ego).square().sum().backward()
        optimizer.step()
    assert torch.equal(adapter(features, torch.zeros(3), ego), features)
    assert not torch.equal(adapter(features, torch.full((3,), .2), ego), features)
    restored = load_delay_adapter(dict(state_dict=adapter.state_dict(), adapter_config=adapter.config()), 8, 'cpu')
    assert torch.equal(restored(features, torch.full((3,), .2), ego), adapter(features, torch.full((3,), .2), ego))
    with pytest.raises(ValueError, match='dimension'):
        load_delay_adapter(dict(state_dict=adapter.state_dict(), adapter_config=adapter.config()), 9, 'cpu')


@pytest.mark.parametrize('two_cam,all_speeds', [(False, True), (True, True), (False, False), (True, False)])
def test_model_variants_and_exact_deployed_distribution(two_cam, all_speeds):
    cfg = config(two_cam, all_speeds)
    model = CameraModel(cfg).eval().requires_grad_(False)
    features = torch.randn(3, model.feature_dim)
    speeds = torch.tensor([0., 3., 20.])
    commands = torch.tensor([0, 3, 5])
    actual = active_logits(model, features, speeds, commands, cfg)
    for i in range(3):
        steer, throttle, brake = model.policy_from_features(features[i:i + 1], int(commands[i]), speeds[i:i + 1])
        if all_speeds:
            pos = float((speeds[i] / 8).clamp(0, 1) * 3)
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            steer, throttle, brake = [(1 - (pos - lo)) * v[lo] + (pos - lo) * v[hi] for v in (steer, throttle, brake)]
        expected = torch.cat([steer.repeat(model.num_throts), throttle.repeat_interleave(model.num_steers), brake[None]])
        assert torch.allclose(actual[i], expected, atol=1e-6)
    with torch.no_grad():
        wide, narrow = torch.rand(2, 3, 32, 48) * 255, torch.rand(2, 3, 32, 32) * 255
        extracted = model.extract_policy_features(wide, narrow)
        output = model(wide, narrow, None if all_speeds else speeds[:2])[0]
        head_output = model.action_logits_from_features(extracted, speeds[:2])
        if not all_speeds:
            head_output = head_output.squeeze(2)
        assert torch.allclose(output, head_output, atol=1e-6)


def test_training_updates_only_adapter_and_preserves_bn(tmp_path):
    make_sequence(tmp_path, 'one')
    cfg = config()
    ds = DelayedJSONActionDataset(discover_sequences(tmp_path), cfg, delays_s=[0, .2])
    loader = torch.utils.data.DataLoader(ds, batch_size=3, drop_last=False)
    model = CameraModel(cfg).eval().requires_grad_(False)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = DelayFeatureAdapter(model.feature_dim, ego_dim=5, max_latency_s=.2)
    before = {k: v.clone() for k, v in adapter.state_dict().items()}
    args = SimpleNamespace(feature_weight=1., temperature=1., kl_direction='student_teacher', log_every=0)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.001)
    metrics = run_epoch(model, adapter, loader, cfg, torch.device('cpu'), args, optimizer)
    assert sum(row['samples'] for row in metrics.values()) == len(ds)  # includes partial final batch
    assert all(torch.equal(frozen[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert any(not torch.equal(before[k], v) for k, v in adapter.state_dict().items())
    assert metrics['0ms']['action_kl'] == pytest.approx(0, abs=1e-7)
    assert metrics['0ms']['feature_mse'] == 0


def test_rig_requires_exact_frame_and_matches_sensor_timestamp():
    rig = CameraRig.__new__(CameraRig)
    rig.queues = {'Wide_RGB_0': queue.Queue(), 'Narrow_RGB': queue.Queue()}
    def message(frame, stamp):
        return SimpleNamespace(frame=frame, timestamp=stamp, raw_data=bytes([1, 2, 3, 255]), height=1, width=1)
    rig.queues['Wide_RGB_0'].put((message(9, .45), 10.))
    for inbox in rig.queues.values():
        inbox.put((message(10, .5), 11.))
    images, timestamp, arrival = rig.read(10)
    assert timestamp == .5 and arrival == 11 and images['Narrow_RGB'][0] == 10
    rig.queues['Wide_RGB_0'].put((message(12, .6), 12.))
    with pytest.raises(RuntimeError, match='skipped'):
        rig.read(11)


def test_route_identity_does_not_depend_on_xml_id(tmp_path):
    path = tmp_path / 'routes.xml'
    path.write_text('<routes><route id="1" town="Town01"><waypoint x="0" y="0"/><waypoint x="2" y="3"/></route>'
                    '<route id="99" town="Town01"><waypoint x="0" y="0"/><waypoint x="2" y="3"/></route></routes>')
    routes = read_routes(path)
    assert routes[0]['route_id'] == routes[1]['route_id']
    with pytest.raises(ValueError, match='Unknown'):
        read_routes(path, ['missing'])


def test_runtime_stale_images_live_ego_and_checkpoint_metadata(tmp_path, monkeypatch):
    """Exercise ImageAgent itself without requiring the CARLA binary on the test host."""
    import importlib.util
    import sys
    import types
    import yaml
    import enum
    class RoadOption(enum.Enum):
        LANEFOLLOW = 4
    class BaseAgent:
        def __init__(self, path):
            self.setup(path)
            self._global_plan = []
    class Waypointer:
        def __init__(self, *_):
            pass
        def tick(self, _):
            return 0, 0, RoadOption.LANEFOLLOW
    for name, values in {
        'carla': dict(VehicleControl=lambda **kw: SimpleNamespace(**kw)),
        'leaderboard_codes.autonomous_agent1': dict(AutonomousAgent=BaseAgent, Track=SimpleNamespace(SENSORS='sensors')),
        'leaderboard_codes.local_planner': dict(RoadOption=RoadOption),
        'pcla_agents.wor.waypointer': dict(Waypointer=Waypointer),
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('runtime_agent_test', Path(__file__).parents[1] / 'pcla_agents/wor/image_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class TinyModel(torch.nn.Module):
        feature_dim = 4
        def __init__(self, _):
            super().__init__()
        def policy(self, wide, narrow, cmd, **kwargs):
            self.last = dict(wide=wide.clone(), narrow=narrow.clone(), command=cmd, **kwargs)
            return torch.zeros(4, 3), torch.zeros(4, 2), torch.zeros(4)
    monkeypatch.setattr(module, 'CameraModel', TinyModel)
    model_path, adapter_path = tmp_path / 'model.th', tmp_path / 'adapter.th'
    torch.save({}, model_path)
    adapter = DelayFeatureAdapter(4, hidden_dim=7, max_latency_s=.2, correction_scale=.3, ego_dim=5)
    torch.save(dict(state_dict=adapter.state_dict(), adapter_config=adapter.config()), adapter_path)
    cfg = dict(config(), main_model_dir=str(model_path), time_adapter_path=str(adapter_path),
               time_parameterized_control=True, observation_latency_s=.1, sensor_tick=.05,
               max_steers=1., max_throts=1., device='cpu')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    agent = module.ImageAgent(str(path))
    assert agent.delay_adapter.hidden_dim == 7 and agent.delay_adapter.correction_scale == .3
    vehicle = SimpleNamespace(get_control=lambda: SimpleNamespace(steer=.1, throttle=.2, brake=.3),
                              get_angular_velocity=lambda: SimpleNamespace(z=90.))
    for i in range(5):
        packet = {f'Wide_RGB_{j}': (i, np.full((32, 16, 4), i * 10, dtype=np.uint8)) for j in range(3)}
        packet.update(Narrow_RGB=(i, np.full((32, 32, 4), i * 10, dtype=np.uint8)),
                      EGO=(i, dict(speed=float(i + 1))), GPS=(i, np.zeros(3)))
        agent.run_step(packet, i * .05, vehicle, camera_timestamp_s=i * .05, received_monotonic_s=100 + i)
    last = agent.image_model.last
    assert agent.latest_timing['camera_frame'] == 2
    assert agent.latest_timing['observation_age_sim_s'] == pytest.approx(.1)
    assert agent.latest_timing['received_monotonic_s'] == 102
    assert not agent.latest_timing['warmup']
    assert last['wide'].shape == (1, 3, 32, 48) and torch.all(last['wide'] == 20)
    assert last['ego_state'].tolist()[0] == pytest.approx([5, np.pi / 2, .1, .2, .3])
    with pytest.raises(ValueError, match='training range'):
        agent.set_observation_latency(.25)
    agent.set_observation_latency(0)
    agent.run_step(packet, .2, vehicle, camera_timestamp_s=.2, received_monotonic_s=104)
    assert agent.latest_timing['camera_frame'] == 4
    assert len(agent.frame_history) == 5  # duplicate sensor packet was not appended
    packet['Narrow_RGB'] = (99, packet['Narrow_RGB'][1])
    with pytest.raises(ValueError, match='frame-aligned'):
        agent.run_step(packet, .2, vehicle, camera_timestamp_s=.2)
    agent.destroy()
    assert not agent.frame_history


def test_camera_calibration_mismatch_rejected(tmp_path):
    make_sequence(tmp_path, 'one')
    sequences = discover_sequences(tmp_path)
    sequences[0]['camera_config'] = dict(camera_x=9.)
    with pytest.raises(ValueError, match='camera mismatch'):
        DelayedJSONActionDataset(sequences, config())


def test_original_adapter_checkpoint_remains_loadable():
    adapter = DelayFeatureAdapter(feature_dim=512, max_latency_s=.25)
    restored = load_delay_adapter(dict(state_dict=adapter.state_dict(), feature_dim=512, max_latency_s=.25), 512, 'cpu')
    assert restored.ego_dim == 0 and restored.max_latency_s == .25
    features = torch.randn(2, 512)
    assert torch.equal(restored(features, torch.tensor([0., .2])), features)
