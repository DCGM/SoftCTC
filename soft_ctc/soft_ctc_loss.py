import torch

from soft_ctc.models.batch_connections import BatchConnections


class SoftCTCLoss(torch.autograd.Function):
    def __init__(self, norm_step=10, zero_infinity=True):
        self._norm_step = norm_step
        self._zero_infinity = zero_infinity

    def __call__(self, logits, connections: BatchConnections, labels, label_probs):
        return self.apply(logits, connections, labels, label_probs, self._norm_step, self._zero_infinity)

    @staticmethod
    def forward(ctx, logits, connections: BatchConnections, labels, label_probs, norm_step=10, zero_infinity=False):
        full_probs = torch.nn.functional.softmax(logits, dim=1)
        full_probs[full_probs == 0] = 1e-37

        N, C, T = full_probs.shape
        L = connections.size()

        alphas = torch.zeros((N, L, T), dtype=logits.dtype, device=connections.device())
        probs = torch.gather(full_probs, 1, labels.unsqueeze(-1).expand(-1, -1, T))

        alphas[:, :, 0] = connections.forward_start * probs[:, :, 0]

        c = torch.sum(alphas[:, :, 0], dim=1)
        alphas[:, :, 0] /= c.view(N, 1)
        ll_forward = torch.log(c)

        current_vector = alphas[:, :, 0]
        for t in range(1, T):
            current_vector = torch.bmm(current_vector.reshape(N, 1, -1), connections.forward).reshape(N, -1) * probs[:, :, t]

            if t % norm_step == 0:
                c = torch.sum(current_vector, dim=1)
                current_vector /= c.view(N, 1)
                ll_forward += torch.log(c)

            alphas[:, :, t] = current_vector

        c = torch.sum(alphas[:, :, -1] * connections.forward_end, dim=1)
        alphas[:, :, -1] /= c.view(N, 1)
        ll_forward += torch.log(c)

        zero_mask = None
        if zero_infinity:
            zero_mask = torch.isinf(ll_forward)
            ll_forward[zero_mask] = 0

        ctx.save_for_backward(logits, labels, label_probs)
        ctx.connections = connections
        ctx.full_probs = full_probs
        ctx.probs = probs
        ctx.alphas = alphas
        ctx.norm_step = norm_step
        ctx.zero_infinity = zero_infinity
        ctx.zero_mask = zero_mask

        return -ll_forward

    @staticmethod
    def backward(ctx, ll_forward):
        logits, labels, label_probs = ctx.saved_tensors
        connections = ctx.connections
        full_probs = ctx.full_probs
        probs = ctx.probs
        alphas = ctx.alphas
        norm_step = ctx.norm_step
        zero_infinity = ctx.zero_infinity
        zero_mask = ctx.zero_mask

        N, C, T = full_probs.shape
        L = connections.size()

        label_probs[label_probs == 0] = 1e-37
        label_probs = torch.tile(label_probs, (1, 1, T))

        betas = torch.zeros_like(alphas)

        betas[:, :, -1] = connections.backward_start * probs[:, :, -1]

        c = torch.sum(betas[:, :, -1], dim=1)
        betas[:, :, -1] /= c.view(N, 1)
        ll_backward = torch.log(c)

        current_vector = betas[:, :, -1]
        for t in range(T - 2, -1, -1):
            current_vector = torch.bmm(current_vector.reshape(N, 1, -1), connections.backward).reshape(N, -1) * probs[:, :, t]

            if t % norm_step == 0:
                c = torch.sum(current_vector, dim=1)
                current_vector /= c.view(N, 1)
                ll_backward += torch.log(c)

            betas[:, :, t] = current_vector

        c = torch.sum(betas[:, :, 0] * connections.backward_end, dim=1)
        betas[:, :, 0] /= c.view(N, 1)
        ll_backward += torch.log(c)

        # ll_diff = torch.abs(ll_forward - ll_backward)
        # if ll_diff > 1e-5:
        #     print(f"Diff in forward/backward LL : abs({ll_forward.item()} - {ll_backward.item()}) = {ll_diff.item()}")

        grad = torch.zeros_like(logits)
        ab = alphas * betas
        ab /= label_probs

        reshaped_labels = torch.tile(labels.reshape(N, L, 1), (1, 1, T))
        grad.scatter_add_(1, reshaped_labels, ab)

        ab /= probs

        ab_sum = torch.sum(ab, dim=1, keepdim=True)

        denominator = full_probs * ab_sum
        denominator[denominator == 0] = 1e-37

        grad = full_probs - grad / denominator

        if zero_infinity:
            grad[zero_mask] = 0
            for n in range(N):
                if torch.any(torch.isinf(grad[n])) or torch.any(torch.isnan(grad[n])):
                    grad[n] = 0

        del ctx.connections
        del ctx.full_probs
        del ctx.probs
        del ctx.alphas
        del ctx.norm_step
        del ctx.zero_infinity

        return grad, None, None, None, None, None
