"""3D U-Net 主干：(2+1)D ResBlock + factorized spatial/temporal attention。"""
from .unet3d import UNet3D, UNet3DConfig

__all__ = ["UNet3D", "UNet3DConfig"]
