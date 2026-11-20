"""Model registry.

The keys of MODELS are the values of --model. They match the names used in the original
RETFound models_vit.py, so checkpoints trained there (whose saved args.model is one of these)
can be rebuilt here with build_model(ckpt["args"]).
"""
from .dinov2 import _DINOV2_ARCH, Dinov2, RETFound_dinov2
from .dinov3 import Dinov3
from .hybrid import Hybrid, SwinEfficientNetHybrid
from .pretrained import load_pretrained
from .retfound_mae import RETFound_mae, VisionTransformer
from .swin_ms import SwinMS, SwinMultiStage
from .timm_model import Timm

MODELS = {
    "RETFound_mae": RETFound_mae,
    "RETFound_dinov2": RETFound_dinov2,
    "Dinov2": Dinov2,
    "Dinov3": Dinov3,
    "Timm": Timm,
    "Hybrid": Hybrid,
    "SwinMS": SwinMS,
}


# What --model_arch means for each --model (shown by `python train_dr.py --help`)
MODEL_ARCH_HELP = {
    "Timm": "REQUIRED: any timm model name, e.g. swinv2_large_window12to24_192to384, "
            "swinv2_base_window12to24_192to384, convnext_base, efficientnet_b3",
    "Dinov2": f"REQUIRED: one of {', '.join(_DINOV2_ARCH)}  (+ --finetune <local .pth>)",
    "Dinov3": "REQUIRED: torch.hub entry, e.g. dinov3_vits16, dinov3_vitl16  (+ --finetune <local .pth>)",
    "RETFound_dinov2": "ignored (fixed ViT-L/14); weights via --finetune RETFound_dinov2_shanghai / RETFound_dinov2_meh",
    "RETFound_mae": "ignored (fixed ViT-L/16); weights via --finetune RETFound_mae_natureCFP / _meh / _shanghai / _natureOCT",
    "Hybrid": "ignored (fixed EfficientNet-B0 + Swin V2-T, torchvision ImageNet weights)",
    "SwinMS": "REQUIRED: timm Swin V2 name, e.g. swinv2_large_window12to24_192to384  (+ --ms_stages / --ms_head_dropout)",
}


def build_model(args, drop_path_rate=None):
    """Build the model named by args.model (no --finetune weights; see load_pretrained).

    drop_path_rate defaults to args.drop_path; pass 0.0 for inference.
    """
    if args.model not in MODELS:
        raise ValueError(f"Unknown --model '{args.model}'. Expected one of: {', '.join(MODELS)}")
    dp = args.drop_path if drop_path_rate is None else drop_path_rate
    if args.model == "RETFound_mae":
        return RETFound_mae(img_size=args.input_size, num_classes=args.nb_classes,
                            drop_path_rate=dp, global_pool=args.global_pool)
    return MODELS[args.model](num_classes=args.nb_classes, drop_path_rate=dp, args=args)


__all__ = ["MODELS", "MODEL_ARCH_HELP", "build_model", "load_pretrained", *MODELS, "SwinEfficientNetHybrid", "SwinMultiStage", "VisionTransformer"]
