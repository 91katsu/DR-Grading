"""Any timm model with its own ImageNet weights, selected by --model_arch."""
import timm


def Timm(args, **kwargs):
    """e.g. the deployed DR model: --model_arch swinv2_large_window12to24_192to384 --input_size 384
    (timm weights: ms_in22k_ft_in1k). Also works for convnext_*, efficientnet_*, resnet*, ...
    Pretrained weights are loaded here, so --finetune is not needed.
    """
    model = timm.create_model(args.model_arch, pretrained=True, **kwargs)
    return model
