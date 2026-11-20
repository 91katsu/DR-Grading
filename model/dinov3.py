"""DINOv3 backbone from torch.hub + a fresh linear head."""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def Dinov3(args, **kwargs):
    # Load the backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=args.model_arch,
        pretrained=False,   # weights are loaded by model.pretrained.load_pretrained (--finetune)
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model
