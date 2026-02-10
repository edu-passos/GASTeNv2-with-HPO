import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import GaussianNLLLoss, KLDivLoss, BCEWithLogitsLoss


from contextlib import contextmanager
from src.utils.min_norm_solvers import MinNormSolver
from src.utils.classifier_io import as_pos_prob


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def no_param_grads(module):
    """
    Temporarily disable requires_grad for all parameters of a module.
    Keeps correct gradients w.r.t. inputs (fake images), avoids allocating
    grads for module parameters, and freezes train-time state updates
    (e.g., BatchNorm running stats) during generator updates.
    """
    req = [p.requires_grad for p in module.parameters()]
    was_training = module.training
    try:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(module.parameters(), req):
            p.requires_grad_(r)
        module.train(was_training)


def grads_as_list(params):
    """
    Return gradients as a list in fixed parameter order, filling missing grads with zeros.
    Required for MGDA to be well-defined.
    """
    out = []
    for p in params:
        if p.grad is None:
            out.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            out.append(p.grad.detach().clone())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────
class UpdateGenerator:
    def __init__(self, crit):
        self.crit = crit  # MUST remain the GAN generator criterion

    def __call__(self, G, D, optim, noise, device):
        raise NotImplementedError

    def get_loss_terms(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla GAN (Step-1)
# ─────────────────────────────────────────────────────────────────────────────
class UpdateGeneratorGAN(UpdateGenerator):
    def __init__(self, crit):
        super().__init__(crit)

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        optim.zero_grad(set_to_none=True)

        with no_param_grads(D):
            fake_data = G(noise)
            d_out = D(fake_data)
            loss = self.crit(device, d_out)

        loss.backward()
        optim.step()
        return loss, {}

    def get_loss_terms(self):
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GASTeN (abs distance to 0.5) — kept for completeness
# ─────────────────────────────────────────────────────────────────────────────
class UpdateGeneratorGASTEN(UpdateGenerator):
    def __init__(self, crit, C, alpha):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        optim.zero_grad(set_to_none=True)

        with no_param_grads(D), no_param_grads(self.C):
            fake_data = G(noise)
            loss_adv = self.crit(device, D(fake_data))
            p = as_pos_prob(self.C(fake_data))
            loss_conf = (0.5 - p).abs().mean()
            loss = loss_adv + self.alpha * loss_conf

        loss.backward()
        optim.step()
        return loss, {"original_g_loss": loss_adv.item(), "conf_dist_loss": loss_conf.item()}

    def get_loss_terms(self):
        return ["original_g_loss", "conf_dist_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# GASTeN + MGDA — corrected gradient alignment
# ─────────────────────────────────────────────────────────────────────────────
class UpdateGeneratorGASTEN_MGDA(UpdateGenerator):
    def __init__(self, crit, C, alpha=1, normalize=False):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.normalize = normalize

    def gradient_normalizers(self, grads, loss):
        return loss.item() * np.sqrt(np.sum([g.pow(2).sum().item() for g in grads]))

    def __call__(self, G, D, optim, noise, device):
        params = [p for p in G.parameters() if p.requires_grad]

        # Objective 1: GAN adversarial
        G.zero_grad()
        optim.zero_grad(set_to_none=True)
        with no_param_grads(D), no_param_grads(self.C):
            fake1 = G(noise)
            term_1 = self.crit(device, D(fake1))
        term_1.backward()
        g1 = grads_as_list(params)

        # Objective 2: classifier ambiguity
        G.zero_grad()
        optim.zero_grad(set_to_none=True)
        with no_param_grads(D), no_param_grads(self.C):
            fake2 = G(noise)
            p = as_pos_prob(self.C(fake2))
            term_2 = (0.5 - p).abs().mean()
        term_2.backward()
        g2 = grads_as_list(params)

        if self.normalize:
            gn1 = self.gradient_normalizers(g1, term_1)
            gn2 = self.gradient_normalizers(g2, term_2)
            g1 = [gr / (gn1 + 1e-12) for gr in g1]
            g2 = [gr / (gn2 + 1e-12) for gr in g2]

        scale, _ = MinNormSolver.find_min_norm_element([g1, g2])

        # Combined update
        G.zero_grad()
        optim.zero_grad(set_to_none=True)
        with no_param_grads(D), no_param_grads(self.C):
            fake = G(noise)
            t1 = self.crit(device, D(fake))
            p = as_pos_prob(self.C(fake))
            t2 = (0.5 - p).abs().mean()
            loss = scale[0] * t1 + scale[1] * t2

        loss.backward()
        optim.step()
        return loss, {
            "original_g_loss": t1.item(),
            "conf_dist_loss": t2.item(),
            "scale1": float(scale[0]),
            "scale2": float(scale[1]),
        }

    def get_loss_terms(self):
        return ["original_g_loss", "conf_dist_loss", "scale1", "scale2"]


# ─────────────────────────────────────────────────────────────────────────────
# GASTeN v2 — Gaussian ambiguity loss (single-step, correct adversarial term)
# ─────────────────────────────────────────────────────────────────────────────
class UpdateGeneratorGASTEN_gaussian(UpdateGenerator):
    def __init__(self, crit, C, alpha, var):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss(reduction="mean")
        self.target = 0.5

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        optim.zero_grad(set_to_none=True)

        with no_param_grads(D), no_param_grads(self.C):
            fake = G(noise)

            # Classifier ambiguity on P(pos)
            p = as_pos_prob(self.C(fake))
            tgt = torch.full_like(p, fill_value=self.target, device=device)
            var = torch.full_like(p, fill_value=self.var, device=device)
            loss_conf = self.c_loss(p, tgt, var)

            # GAN adversarial
            loss_adv = self.crit(device, D(fake))
            loss = loss_adv + self.alpha * loss_conf

        loss.backward()
        clip_grad_norm_(G.parameters(), 0.50)
        optim.step()

        return loss, {"original_g_loss": loss_adv.item(), "conf_dist_loss": loss_conf.item()}

    def get_loss_terms(self):
        return ["original_g_loss", "conf_dist_loss"]


class UpdateGeneratorGASTEN_gaussianV2(UpdateGenerator):
    """
    Uses C(fake, output_feature_maps=True); prediction is still converted to P(pos).
    """
    def __init__(self, crit, C, alpha, var):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss(reduction="mean")
        self.target = 0.5

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        optim.zero_grad(set_to_none=True)

        with no_param_grads(D), no_param_grads(self.C):
            fake = G(noise)

            out = self.C(fake, output_feature_maps=True)
            p = as_pos_prob(out)

            tgt = torch.full_like(p, fill_value=self.target, device=device)
            var = torch.full_like(p, fill_value=self.var, device=device)
            loss_conf = self.c_loss(p, tgt, var)

            loss_adv = self.crit(device, D(fake))
            loss = loss_adv + self.alpha * loss_conf

        loss.backward()
        clip_grad_norm_(G.parameters(), 0.50)
        optim.step()

        return loss, {"original_g_loss": loss_adv.item(), "conf_dist_loss": loss_conf.item()}

    def get_loss_terms(self):
        return ["original_g_loss", "conf_dist_loss"]
