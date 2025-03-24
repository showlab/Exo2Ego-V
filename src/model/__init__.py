"""
Reference: https://github.com/sxyu/pixel-nerf
"""

from .models import PixelNeRFNet2


def make_model(conf,  *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf2")  # single
    if model_type == "pixelnerf2":
        net = PixelNeRFNet2(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
