"""Image transforms: CLAHE, timm default augmentation, fundus-specific augmentation."""
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CLAHE:
    """Same as the deployed DR model (deployment/DR_cls/classification/clahe.py):
    CLAHE (clip 4.0, 8x8 tiles) on the Y channel of YUV, PIL RGB in -> PIL RGB out."""

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit, self.tile_grid_size = clip_limit, tile_grid_size

    def __call__(self, img):
        import cv2
        import numpy as np
        from PIL import Image
        yuv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        yuv[..., 0] = clahe.apply(np.ascontiguousarray(yuv[..., 0]))
        return Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))

    def __repr__(self):
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"


def build_transform(is_train, args):
    """is_train: 'train' | 'val' | 'test'. --clahe is applied before everything, on every split."""
    t = _build_transform(is_train, args)
    if getattr(args, "clahe", False):
        t = transforms.Compose([CLAHE(), t])
    return t


def _build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # "fundus": milder, colour-preserving augmentation for pre-cropped square fundus images
    if getattr(args, "aug_mode", "default") == "fundus":
        return _fundus_transform(is_train, args, mean, std)

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)


def _fundus_transform(is_train, args, mean, std):
    """Augmentation for DR grading on fundus images already cropped to the retinal disc.

    vs. the timm/ImageNet default:
      * crop keeps 70-100% of the area (default goes down to 8%, which can cut away the lesions
        that define the grade while the label stays the same)
      * no Invert / Solarize / Posterize / Equalize; only mild brightness/contrast/saturation,
        because lesion colour (red haemorrhage/MA, yellow exudate) is diagnostic
      * vertical flip + rotation (fundus has no canonical up/down at this level)
      * no random erasing (could hide the only lesion)
      * eval resizes the whole image to input_size (no centre crop that clips the periphery)
    """
    size = args.input_size
    interp = transforms.InterpolationMode.BICUBIC
    if is_train == 'train':
        t = [
            transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=interp),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    else:
        t = [transforms.Resize((size, size), interpolation=interp)]
    t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(t)
