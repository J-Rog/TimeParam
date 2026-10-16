#!/usr/bin/env python3
"""Self-distill current frozen WoR features/actions into a stale-image adapter.

Only DelayFeatureAdapter is trained. No act0/act_0, Q-values, segmentation labels,
CARLA installation, or original WoR dataset are needed. See docs/SELF_DISTILLATION.md.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from delay_dataset import DelayedJSONActionDataset, discover_sequences, split_sequences
from pcla_agents.wor.rails.models.main_model import CameraModel, action_logits
from pcla_agents.wor.rails.models.time_adapter import DelayFeatureAdapter

# Retain helper names for users of the previous trainer.
combine_action_logits = action_logits


def action_logits_from_features(base_model, features, speeds):
    return base_model.action_logits_from_features(features, speeds)


def load_base_model(config_path, checkpoint_path, device):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    config['imagenet_pretrained'] = False
    model = CameraModel(config).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload.get('state_dict', payload), strict=True)
    model.eval().requires_grad_(False)
    return model, config


def active_logits(model, features, speeds, commands, config):
    """Match runtime command selection and linear interpolation at live ego speed."""
    logits = model.driving_logits_from_features(features, speeds)
    logits = logits[torch.arange(len(commands), device=commands.device), commands]
    if not model.all_speeds:
        return logits[:, 0]
    span = float(config['max_speeds']) - float(config['min_speeds'])
    if span <= 0:
        raise ValueError('max_speeds must exceed min_speeds')
    position = ((speeds - config['min_speeds']) / span * (model.num_speeds - 1)).clamp(0, model.num_speeds - 1)
    lower, upper = position.floor().long(), position.ceil().long()
    rows = torch.arange(len(speeds), device=speeds.device)
    weight = (position - lower)[:, None]
    return (1 - weight) * logits[rows, lower] + weight * logits[rows, upper]


def distillation_loss(student_logits, teacher_logits, features, teacher_features,
                      feature_weight=1.0, temperature=1.0, direction='student_teacher'):
    if temperature <= 0 or feature_weight < 0:
        raise ValueError('Positive temperature and nonnegative feature weight required')
    student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    if direction == 'student_teacher':
        # Explicit KL(student || teacher), not torch.kl_div's reverse argument order.
        kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1).mean()
    elif direction == 'teacher_student':
        kl = F.kl_div(student_logp, teacher_logp, reduction='batchmean', log_target=True)
    else:
        raise ValueError(f'Unknown KL direction: {direction}')
    kl = kl * temperature ** 2
    feature_loss = F.mse_loss(features, teacher_features.detach())
    return kl + feature_weight * feature_loss, kl, feature_loss


def run_epoch(model, adapter, loader, config, device, args, optimizer=None):
    model.eval()  # Keep BN buffers frozen as well as all driver weights.
    adapter.train(optimizer is not None)
    totals, counts = {}, {}
    for batch_index, batch in enumerate(loader):
        images = {key: batch[key].to(device, dtype=torch.float32).permute(0, 3, 1, 2)
                  for key in ('stale_wide', 'stale_narr', 'current_wide', 'current_narr')}
        speeds = batch['speed'].to(device, dtype=torch.float32)
        commands = batch['command'].to(device)
        latencies = batch['latency_s'].to(device, dtype=torch.float32)
        ego = batch['ego_state'].to(device)
        with torch.no_grad():
            stale = model.extract_policy_features(images['stale_wide'], images['stale_narr'])
            current = model.extract_policy_features(images['current_wide'], images['current_narr'])
            teacher = active_logits(model, current, speeds, commands, config)
        with torch.set_grad_enabled(optimizer is not None):
            corrected = adapter(stale, latencies, ego)
            student = active_logits(model, corrected, speeds, commands, config)
            loss, kl, mse = distillation_loss(student, teacher, corrected, current,
                args.feature_weight, args.temperature, args.kl_direction)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite distillation loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        if args.log_every and (batch_index + 1) % args.log_every == 0:
            phase = 'train' if optimizer is not None else 'val'
            print(f'{phase} batch={batch_index + 1}/{len(loader)} loss={float(loss.detach()):.6f}', flush=True)
        # Offline validation reports both delayed baseline and corrected loss per delay.
        with torch.no_grad():
            baseline = active_logits(model, stale, speeds, commands, config)
            for delay in batch['requested_delay_s'].unique():
                mask = batch['requested_delay_s'].to(device) == delay.to(device)
                n = int(mask.sum())
                name = f'{float(delay) * 1000:g}ms'
                _, dkl, dmse = distillation_loss(student[mask], teacher[mask], corrected[mask], current[mask],
                    args.feature_weight, args.temperature, args.kl_direction)
                _, bkl, bmse = distillation_loss(baseline[mask], teacher[mask], stale[mask], current[mask],
                    args.feature_weight, args.temperature, args.kl_direction)
                values = np.array([float(dkl), float(dmse), float(bkl), float(bmse)])
                totals[name] = totals.get(name, np.zeros(4)) + values * n
                counts[name] = counts.get(name, 0) + n
    return {k: dict(zip(('action_kl', 'feature_mse', 'baseline_action_kl', 'baseline_feature_mse'),
                        (v / counts[k]).tolist()), samples=counts[k]) for k, v in totals.items()}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        raise ValueError('Invalid training size, workers or learning rate')
    if args.temperature <= 0 or args.feature_weight < 0:
        raise ValueError('Positive temperature and nonnegative feature weight required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, config = load_base_model(args.config_path, args.checkpoint, device)
    delays_s = [d / 1000 for d in args.delays_ms]
    sequences = discover_sequences(args.data_dir)
    manifest = json.loads(Path(args.split_manifest).read_text()) if args.split_manifest else None
    training, validation, split = split_sequences(sequences, args.seed, args.val_fraction,
                                                 args.heldout_towns, manifest)
    ego_dim = {'none': 0, 'speed': 1, 'full': 5}[args.ego_state]
    adapter = DelayFeatureAdapter(model.feature_dim, args.hidden_dim, args.max_latency_s,
                                  args.correction_scale, ego_dim).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    loaders = [DataLoader(DelayedJSONActionDataset(group, config, delays_s, args.max_latency_s, ego_dim),
                          batch_size=args.batch_size, shuffle=is_train, num_workers=args.num_workers,
                          pin_memory=device.type == 'cuda', drop_last=False)
               for group, is_train in ((training, True), (validation, False))]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.split.json').write_text(json.dumps(split, indent=2) + '\n')
    teacher_hash = file_sha256(args.checkpoint)
    best, history = float('inf'), []
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, adapter, loaders[0], config, device, args, optimizer)
        val_metrics = run_epoch(model, adapter, loaders[1], config, device, args)
        score = sum(m['action_kl'] + args.feature_weight * m['feature_mse'] for m in val_metrics.values()) / len(val_metrics)
        history.append(dict(epoch=epoch + 1, train=train_metrics, val=val_metrics, val_loss=score))
        print(json.dumps(history[-1]), flush=True)
        if score < best:
            best = score
            payload = dict(format_version=2, objective='self_distilled_delay', state_dict=adapter.state_dict(),
                           policy_distribution='image_agent_legacy',
                           adapter_config=adapter.config(), epoch=epoch + 1, val_loss=score,
                           base_checkpoint_sha256=teacher_hash, model_config=config,
                           delays_s=delays_s, split=split, train_routes=sorted({s['route'] for s in training}),
                           val_routes=sorted({s['route'] for s in validation}),
                           temperature=args.temperature, kl_direction=args.kl_direction,
                           feature_weight=args.feature_weight, seed=args.seed)
            temp = output.with_suffix(output.suffix + '.tmp')
            torch.save(payload, temp)
            temp.replace(output)
            print(f'Saved best adapter: {output}', flush=True)
        output.with_suffix('.metrics.json').write_text(json.dumps(history, indent=2) + '\n')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--config-path', default='outputs/config_timeparam.yaml')
    p.add_argument('--checkpoint', default='pcla_agents/wor_pretrained/nocrash_weights/main_model_16.th')
    p.add_argument('--output', default='outputs/time_adapter.th')
    p.add_argument('--delays-ms', type=float, nargs='+', default=[0, 50, 100, 150, 200])
    p.add_argument('--max-latency-s', type=float, default=.2)
    p.add_argument('--ego-state', choices=['none', 'speed', 'full'], default='full')
    p.add_argument('--feature-weight', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=1.0, help='Policy distillation temperature; not the old Q-label temperature')
    p.add_argument('--kl-direction', choices=['student_teacher', 'teacher_student'], default='student_teacher')
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--correction-scale', type=float, default=.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--heldout-towns', nargs='*', default=[])
    p.add_argument('--split-manifest')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


if __name__ == '__main__':
    train(parse_args())
