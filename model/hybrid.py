"""EfficientNet-B0 + Swin V2-T concatenated-feature model (torchvision)."""
import torch
import torch.nn as nn


class SwinEfficientNetHybrid(nn.Module):
    """EfficientNet-B0 (1280-d) + Swin V2-T (768-d) features concatenated -> MLP -> classes.

    Same structure and parameter names as the deployed
    deployment/DR_cls/classification/model.py::SwinEfficientNetHybridModel
    (efficientnet.*, swin.*, mlp.*), so its checkpoints load with load_state_dict.
    Difference: both backbones start from torchvision ImageNet-1k weights (deployed code: weights=None,
    then loads its own fine-tuned .pt).
    """

    def __init__(self, num_classes=5, drop_path_rate=0.2, pretrained=True, mlp_dropout=0.0):
        super().__init__()
        from torchvision.models import (efficientnet_b0, swin_v2_t,
                                        EfficientNet_B0_Weights, Swin_V2_T_Weights)
        from torchvision.ops import StochasticDepth

        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.efficientnet = nn.Sequential(*list(eff.children())[:-1])  # features + avgpool -> (B,1280,1,1)

        self.swin = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.swin.head = nn.Identity()  # -> (B,768)
        # torchvision fixes the stochastic depth of both backbones at 0.2 (linearly increasing per block);
        # rescale every block's drop prob so that --drop_path plays the role of that 0.2
        for mod in list(self.efficientnet.modules()) + list(self.swin.modules()):
            if isinstance(mod, StochasticDepth):
                mod.p = mod.p * drop_path_rate / 0.2

        self.num_features = 1280 + 768
        layers = [nn.Linear(self.num_features, 512), nn.ReLU()]
        if mlp_dropout > 0:
            layers.append(nn.Dropout(mlp_dropout))
        layers.append(nn.Linear(512, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward_features(self, x):
        e = torch.flatten(self.efficientnet(x), 1)
        s = self.swin(x)
        return torch.cat((e, s), dim=1)

    def forward_head(self, x, pre_logits=False):
        # pre_logits=True -> the 2048-d concatenated feature (used by tools/svm_dr.py)
        return x if pre_logits else self.mlp(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    @property
    def head(self):  # tools/svm_dr.py calls model.head(feature) to get logits
        return self.mlp


def Hybrid(args, num_classes=5, drop_path_rate=0.2, pretrained=True, **kwargs):
    """--model Hybrid. Swin V2-T was pre-trained at 256; input_size should be a multiple of 32 (e.g. 256 or 384)."""
    return SwinEfficientNetHybrid(num_classes=num_classes, drop_path_rate=drop_path_rate, pretrained=pretrained)
