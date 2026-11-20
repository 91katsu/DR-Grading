"""Load foundation-model pre-trained weights (--finetune) into a freshly built model."""
import torch
from timm.models.layers import trunc_normal_

from utils.pos_embed import interpolate_pos_embed


def load_pretrained(model, args):
    """Mirrors the original RETFound main_finetune.py logic.

    Timm / SwinMS / Hybrid load their ImageNet weights in the constructor; the others load --finetune:
      Dinov2 / Dinov3          : local checkpoint path
      RETFound_dinov2 / _mae   : Hugging Face repo YukunZhou/<finetune>
    """
    if args.model in ("Timm", "SwinMS"):
        print(f"[{args.model}] {args.model_arch}: timm pretrained weights loaded in constructor (--finetune ignored)")
        return model
    if args.model == "Hybrid":
        print("[Hybrid] EfficientNet-B0 + Swin V2-T: torchvision ImageNet-1k weights loaded in constructor "
              "(--finetune / --model_arch ignored)")
        return model
    if not args.finetune:
        return model

    print(f"Preparing to load pre-trained weights: {args.finetune}")
    if args.model in ["Dinov3", "Dinov2"]:
        checkpoint_path = args.finetune  # local path
    elif args.model in ["RETFound_dinov2", "RETFound_mae"]:
        from huggingface_hub import hf_hub_download
        print(f"Downloading pre-trained weights from Hugging Face Hub: {args.finetune}")
        checkpoint_path = hf_hub_download(
            repo_id=f"YukunZhou/{args.finetune}",
            filename=f"{args.finetune}.pth",
        )
    else:
        raise ValueError(
            f"Unsupported model '{args.model}'. "
            f"Expected one of: Dinov3, Dinov2, RETFound_dinov2, RETFound_mae"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded pre-trained checkpoint from: {checkpoint_path}")

    if args.model in ["Dinov3", "Dinov2"]:
        checkpoint_model = checkpoint
    elif args.model == "RETFound_dinov2":
        checkpoint_model = checkpoint["teacher"]
    else:  # RETFound_mae
        checkpoint_model = checkpoint["model"]

    # -- Key hygiene
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

    # -- Remove classifier if shape mismatched
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # -- Interpolate pos embed (ViT)
    interpolate_pos_embed(model, checkpoint_model)

    # -- Load backbone weights (non-strict)
    _ = model.load_state_dict(checkpoint_model, strict=False)

    # -- Re-init head
    if hasattr(model, "head") and hasattr(model.head, "weight"):
        trunc_normal_(model.head.weight, std=2e-5)
    return model
