import torch
import yaml

from pcla_agents.wor.rails.models.main_model import CameraModel

with open(
    "pcla_agents/wor_pretrained/nocrash_weights/config_nocrash.yaml"
) as f:
    config = yaml.safe_load(f)

model = CameraModel(config).cuda().eval()

wide = torch.rand(1, 3, 192, 480, device="cuda") * 255.0
narr = torch.rand(1, 3, 224, 384, device="cuda") * 255.0
cmd = 3

with torch.no_grad():
    wrapper_output = model.policy(wide, narr, cmd)

    features = model.extract_policy_features(wide, narr)
    split_output = model.policy_from_features(features, cmd)

for wrapped, split in zip(wrapper_output, split_output):
    torch.testing.assert_close(wrapped, split)

print("Feature/action split check passed.")
