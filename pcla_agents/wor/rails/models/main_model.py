from typing import Optional, Tuple

import torch
from torch import nn
from pcla_agents.wor.common.resnet import resnet18, resnet34
from pcla_agents.wor.common.normalize import Normalize
from pcla_agents.wor.common.segmentation import SegmentationHead

class CameraModel(nn.Module):
    def __init__(self, config, num_cmds=6):
        super().__init__()
        
        # Configs
        self.num_cmds   = num_cmds
        self.num_steers = config['num_steers']
        self.num_throts = config['num_throts']
        self.num_speeds = config['num_speeds']
        self.num_labels = len(config['seg_channels'])
        self.all_speeds = config['all_speeds']
        self.two_cam    = config['use_narr_cam']
        
        self.backbone_wide = resnet34(pretrained=config['imagenet_pretrained'])
        self.seg_head_wide = SegmentationHead(512, self.num_labels+1)
        # Always register narrator modules so TorchScript sees consistent attributes.
        self.backbone_narr = nn.Identity()
        self.seg_head_narr = nn.Identity()
        self.bottleneck_narr = nn.Identity()
        if self.two_cam:
            self.backbone_narr = resnet18(pretrained=config['imagenet_pretrained'])
            self.seg_head_narr = SegmentationHead(512, self.num_labels+1)
            self.bottleneck_narr = nn.Sequential(
                nn.Linear(512,64),
                nn.ReLU(True),
            )

        self.spd_encoder = nn.Identity()
        if self.all_speeds:
            self.num_acts = self.num_cmds*self.num_speeds*(self.num_steers+self.num_throts+1)
        else:
            self.num_acts = self.num_cmds*(self.num_steers+self.num_throts+1)
            self.spd_encoder = nn.Sequential(
                nn.Linear(1,64),
                nn.ReLU(True),
                nn.Linear(64,64),
                nn.ReLU(True),
            )

        self.wide_seg_head = SegmentationHead(512, self.num_labels+1)
        self.act_head = nn.Sequential(
            nn.Linear(512 + (0 if self.all_speeds else 64) + (64 if self.two_cam else 0),256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,self.num_acts),
        )
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(
        self,
        wide_rgb: torch.Tensor,
        narr_rgb: torch.Tensor,
        spd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        assert (self.all_speeds and spd is None) or \
               (not self.all_speeds and spd is not None)

        wide_embed = self.backbone_wide(self.normalize(wide_rgb/255.))
        wide_seg_output = self.seg_head_wide(wide_embed)
        narr_seg_output = wide_seg_output

        if self.two_cam:
            narr_embed = self.backbone_narr(self.normalize(narr_rgb/255.))
            narr_seg_output = self.seg_head_narr(narr_embed)
            embed = torch.cat([
                wide_embed.mean(dim=[2,3]),
                self.bottleneck_narr(narr_embed.mean(dim=[2,3])),
            ], dim=1)
        else:
            embed = wide_embed.mean(dim=[2,3])


        # Action logits
        if self.all_speeds:
            act_output = self.act_head(embed).view(-1,self.num_cmds,self.num_speeds,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts)
        else:
            assert spd is not None
            act_output = self.act_head(torch.cat([embed, self.spd_encoder(spd[:,None])], dim=1)).view(-1,self.num_cmds,1,self.num_steers+self.num_throts+1)
            act_output = action_logits(act_output, self.num_steers, self.num_throts).squeeze(2)

        return act_output, wide_seg_output, narr_seg_output


    def extract_policy_features(self, wide_rgb, narr_rgb):
        """
        Convert one batch of camera images into the pooled visual feature used by
        the driving/action head.

        Returns:
            Tensor of shape [batch_size, feature_dim].
            feature_dim is normally 512 for the current wor_nc configuration.
        """
        wide_embed = self.backbone_wide(
            self.normalize(wide_rgb / 255.0)
        )

        if self.two_cam:
            narr_embed = self.backbone_narr(
                self.normalize(narr_rgb / 255.0)
            )

            features = torch.cat([
                wide_embed.mean(dim=[2, 3]),
                self.bottleneck_narr(
                    narr_embed.mean(dim=[2, 3])
                ),
            ], dim=1)
        else:
            features = wide_embed.mean(dim=[2, 3])

        return features
    
    def policy_from_features(self, features, cmd, spd=None):
        """
        Apply the existing frozen action head to already-computed visual features.

        Args:
            features: pooled visual features, shape [batch_size, feature_dim]
            cmd: integer route command
            spd: vehicle-speed tensor, required only when all_speeds is False

        Returns:
            steer_logits, throt_logits, brake_logits
        """
        if self.all_speeds:
            act_output = self.act_head(features).view(
                -1,
                self.num_cmds,
                self.num_speeds,
                self.num_steers + self.num_throts + 1,
            )

            steer_logits = act_output[
                0, cmd, :, :self.num_steers
            ]
            throt_logits = act_output[
                0,
                cmd,
                :,
                self.num_steers:self.num_steers + self.num_throts,
            ]
            brake_logits = act_output[0, cmd, :, -1]

        else:
            assert spd is not None

            act_input = torch.cat([
                features,
                self.spd_encoder(spd[:, None]),
            ], dim=1)

            act_output = self.act_head(act_input).view(
                -1,
                self.num_cmds,
                1,
                self.num_steers + self.num_throts + 1,
            )

            steer_logits = act_output[
                0, cmd, 0, :self.num_steers
            ]
            throt_logits = act_output[
                0,
                cmd,
                0,
                self.num_steers:self.num_steers + self.num_throts,
            ]
            brake_logits = act_output[0, cmd, 0, -1]

        return steer_logits, throt_logits, brake_logits

    @torch.no_grad()
    def policy(
        self,
        wide_rgb,
        narr_rgb,
        cmd,
        spd=None,
        delay_adapter=None,
        latency_s=None,
    ):
        """
        Inference-only policy call.

        When no delay adapter is supplied, this follows the original WoR policy
        path. When an adapter and latency are supplied, the adapter modifies the
        pooled visual feature before the unchanged action head is used.
        """
        features = self.extract_policy_features(wide_rgb, narr_rgb)

        if delay_adapter is not None and latency_s is not None:
            features = delay_adapter(features, latency_s)

        return self.policy_from_features(features, cmd, spd)

    def action_logits(raw_logits: torch.Tensor, num_steers: int, num_throts: int) -> torch.Tensor:
        
        steer_logits = raw_logits[...,:num_steers]
        throt_logits = raw_logits[...,num_steers:num_steers+num_throts]
        brake_logits = raw_logits[...,-1:]
        
        steer_logits = steer_logits.repeat(1,1,1,num_throts)
        throt_logits = throt_logits.repeat_interleave(num_steers,-1)
        
        act_logits = torch.cat([steer_logits + throt_logits, brake_logits], dim=-1)
        
        return act_logits



# if __name__ == '__main__':
    
#     from collections import namedtuple
#     Config = namedtuple('Config', [
#         'num_steers', 'num_throts', 'num_speeds', 'seg_channels', 'imagenet_pretrained'
#     ])
    
#     config = Config(num_steers=9, num_throts=3, num_speeds=4, seg_channels=[4,6,7,8,10], imagenet_pretrained=True)
#     model = TwoCameraModel(config)
    
#     wide_rgb = torch.zeros((1,3,128,480))
#     narr_rgb = torch.zeros((1,3,64,384))
    
#     act_output, seg_output = model(wide_rgb, narr_rgb)
#     print (act_output.shape, seg_output.shape)
