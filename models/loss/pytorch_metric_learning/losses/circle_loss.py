import torch

from ..distances import CosineSimilarity
from ..reducers import AvgNonZeroReducer
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .generic_pair_loss import GenericPairLoss


class CircleLoss(GenericPairLoss):
    """
    Circle loss for pairwise labels only.

    Args:
    m:  The relaxation factor that controls the radious of the decision boundary.
    gamma: The scale factor that determines the largest scale of each similarity score.

    According to the paper, the suggested default values of m and gamma are:

    Face Recognition: m = 0.25, gamma = 256
    Person Reidentification: m = 0.25, gamma = 128
    Fine-grained Image Retrieval: m = 0.4, gamma = 80

    By default, we set m = 0.4 and gamma = 80
    """

    def __init__(self, m=0.4, gamma=80, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.m = m
        self.gamma = gamma
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.op = 1 + self.m
        self.on = -self.m
        self.delta_p = 1 - self.m
        self.delta_n = self.m
        self.add_to_recordable_attributes(
            list_of_names=["m", "gamma", "op", "on", "delta_p", "delta_n"],
            is_stat=False,
        )

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_mask_bool = pos_mask.bool()
        neg_mask_bool = neg_mask.bool()
        anchor_positive = mat[pos_mask_bool]
        anchor_negative = mat[neg_mask_bool]
        new_mat = torch.zeros_like(mat)

        new_mat[pos_mask_bool] = (
            -self.gamma
            * torch.relu(self.op - anchor_positive.detach())
            * (anchor_positive - self.delta_p)
        )
        new_mat[neg_mask_bool] = (
            self.gamma
            * torch.relu(anchor_negative.detach() - self.on)
            * (anchor_negative - self.delta_n)
        )

        logsumexp_pos = lmu.logsumexp(
            new_mat, keep_mask=pos_mask_bool, add_one=False, dim=1
        )
        logsumexp_neg = lmu.logsumexp(
            new_mat, keep_mask=neg_mask_bool, add_one=False, dim=1
        )

        losses = self.soft_plus(logsumexp_pos + logsumexp_neg)

        zero_rows = torch.where(
            (torch.sum(pos_mask, dim=1) == 0) | (torch.sum(neg_mask, dim=1) == 0)
        )[0]
        final_mask = torch.ones_like(losses)
        final_mask[zero_rows] = 0
        losses = losses * final_mask
        return {
            "loss": {
                "losses": losses,
                "indices": c_f.torch_arange_from_size(new_mat),
                "reduction_type": "element",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()
