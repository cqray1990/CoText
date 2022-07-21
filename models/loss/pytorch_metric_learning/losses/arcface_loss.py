import numpy as np
import torch

from ..utils import common_functions as c_f
from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class ArcFaceLoss(LargeMarginSoftmaxLoss):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, *args, margin=28.6, scale=64, **kwargs):
        super().__init__(*args, margin=margin, scale=scale, **kwargs)

    def init_margin(self):
        self.margin = np.radians(self.margin)

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)
        return torch.cos(angles + self.margin)

    def scale_logits(self, logits, *_):
        return logits * self.scale
