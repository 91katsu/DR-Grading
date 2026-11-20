"""DINOv2 ViTs (timm): natural-image DINOv2 and RETFound-DINOv2."""
import timm

_DINOV2_ARCH = {
    'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
    'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
    'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
    'dinov2_vitg14': 'vit_giant_patch14_dinov2.lvd142m',
}


def Dinov2(args, **kwargs):
    if args.model_arch not in _DINOV2_ARCH:
        raise ValueError(f"Unknown model_arch '{args.model_arch}'. "
                         f"Expected one of: {', '.join(_DINOV2_ARCH)}")
    model = timm.create_model(
        _DINOV2_ARCH[args.model_arch],
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model


def RETFound_dinov2(args, **kwargs):
    # img_size follows --input_size; model.pretrained interpolates the
    # RETFound pos_embed to the matching grid (input_size / 14).
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=getattr(args, "input_size", 224),
        **kwargs
    )
    return model
