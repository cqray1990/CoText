import torch

from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class GenericPairLoss(BaseMetricLossFunction):
    def __init__(self, mat_based_loss, **kwargs):
        super().__init__(**kwargs)
        self.loss_method = (
            self.mat_based_loss if mat_based_loss else self.pair_based_loss
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
#         print(labels.shape,ref_labels.shape)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
#         print(indices_tuple)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        return self.loss_method(mat, indices_tuple)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def pair_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
#         print(pos_pair.shape,neg_pair.shape)
#         print(mat.shape)
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)
