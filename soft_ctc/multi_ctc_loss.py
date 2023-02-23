import torch
from torch.nn.modules.loss import _Loss


class MultiCTCLoss(_Loss):
    def __init__(self, blank=0, zero_infinity=True):
        super().__init__(reduction='none')
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.ctc_loss = torch.nn.CTCLoss(blank=self.blank, reduction='none', zero_infinity=self.zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths, log_weights, occurrences):
        losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        weighted_losses = losses - log_weights
        out = torch.zeros_like(occurrences, dtype=losses.dtype)

        start = 0
        for i, occ in enumerate(occurrences):
            out[i] = -torch.logsumexp(-weighted_losses[start:start + occ], 0)
            start += occ

        return out
