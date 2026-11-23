"""Multi-stage pooled Swin V2 classifier ("design A").

Plain timm Swin V2 classifies from the LAST stage only:
    stage4 (12x12x1536 @384) -> LayerNorm -> global avg pool -> Linear(1536 -> 5)
Here every selected stage is pooled and the vectors are concatenated, so the head also sees the
higher-resolution stage-2/3 features (small lesions such as micro-aneurysms / small haemorrhages):

    stage1  96x96x192  -> avg pool -> LayerNorm(192)  ┐
    stage2  48x48x384  -> avg pool -> LayerNorm(384)  ├ concat (2880 for all 4) -> Dropout -> Linear -> 5
    stage3  24x24x768  -> avg pool -> LayerNorm(768)  │
    stage4  12x12x1536 -> pretrained norm -> avg pool ┘   (identical to the original timm path)

Parameter names of the backbone are kept (patch_embed.*, layers.<i>.blocks.<j>.*, norm.*), so the
hierarchical layer-wise LR decay in utils/lr_decay.py and timm's no_weight_decay() still apply; the
new parameters (stage_norms.*, head.*) get the full LR.
"""
import timm
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def swin_size_kwargs(args):
    """img_size / window_size for timm Swin models, so --input_size other than the arch's default works.

    timm Swin fixes the input size at build time. Every stage's token grid (input_size/4, /8, /16, /32) must be
    divisible by the window, or be smaller than it (then the window is clamped to the grid), e.g.
      384: window 24 (arch default)   448: --window_size 14 or 28   512: --window_size 16 or 32
    Swin V2 pretrained weights load at any window (continuous position bias). Non-Swin archs: no kwargs.
    """
    if not str(getattr(args, "model_arch", "")).startswith("swin"):
        return {}
    kw = {"img_size": args.input_size}
    if getattr(args, "window_size", None):
        kw["window_size"] = args.window_size
    return kw


class SwinMultiStage(nn.Module):
    def __init__(self, arch="swinv2_large_window12to24_192to384", num_classes=5, drop_path_rate=0.2,
                 stages=(0, 1, 2, 3), head_dropout=0.0, pretrained=True, **size_kwargs):
        super().__init__()
        base = timm.create_model(arch, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate,
                                 **size_kwargs)
        # keep the backbone modules under their original names
        self.patch_embed = base.patch_embed
        self.layers = base.layers
        self.norm = base.norm  # final LayerNorm (pretrained), used for the last stage only
        self._nwd = set(base.no_weight_decay()) if hasattr(base, "no_weight_decay") else set()
        self.pretrained_cfg = getattr(base, "pretrained_cfg", {})

        n_stages = len(self.layers)
        self.stages = sorted(int(s) for s in stages)
        if not self.stages or self.stages[0] < 0 or self.stages[-1] >= n_stages:
            raise ValueError(f"stages must be a subset of 0..{n_stages - 1}, got {stages}")
        dims = [base.layers[i].blocks[-1].norm1.normalized_shape[0] for i in range(n_stages)]
        self.stage_dims = [dims[i] for i in self.stages]
        last = n_stages - 1
        # new LayerNorm for every pooled intermediate stage (puts them on a comparable scale)
        self.stage_norms = nn.ModuleDict({str(i): nn.LayerNorm(dims[i]) for i in self.stages if i != last})
        self.num_features = sum(self.stage_dims)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.num_features, num_classes)
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # --- timm-style hooks used by train_dr / lr_decay / svm_dr ---
    def no_weight_decay(self):
        return self._nwd

    def set_grad_checkpointing(self, enable=True):
        for layer in self.layers:
            layer.grad_checkpointing = enable

    def forward_features(self, x):
        """Returns the concatenated pooled feature (B, sum(stage_dims))."""
        x = self.patch_embed(x)  # (B, H, W, C)
        last = len(self.layers) - 1
        pooled = []
        for i, layer in enumerate(self.layers):
            x = layer(x)  # (B, H, W, C_i)
            if i in self.stages:
                if i == last:
                    pooled.append(self.norm(x).mean(dim=(1, 2)))
                else:
                    pooled.append(self.stage_norms[str(i)](x.mean(dim=(1, 2))))
            if i >= self.stages[-1]:
                break  # later stages are not needed
        return torch.cat(pooled, dim=1)

    def forward_head(self, x, pre_logits=False):
        return x if pre_logits else self.head(self.head_drop(x))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def SwinMS(args, num_classes=5, drop_path_rate=0.2, **kwargs):
    """--model SwinMS --model_arch <timm swinv2 name> [--ms_stages 0 1 2 3]"""
    return SwinMultiStage(arch=args.model_arch, num_classes=num_classes, drop_path_rate=drop_path_rate,
                          stages=getattr(args, "ms_stages", None) or (0, 1, 2, 3),
                          head_dropout=getattr(args, "ms_head_dropout", 0.0), **swin_size_kwargs(args))
