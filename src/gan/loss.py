import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def valid_loss(config):
    valid_names = {'wgan-gp', 'ns', 'hinge-r1'}
    if config["name"].lower() not in valid_names:
        return False

    if config["name"].lower() == "wgan-gp":
        return "args" in config and "lambda" in config["args"]

    if config["name"].lower() == "hinge-r1":
        return "args" in config and "lambda" in config["args"]

    return True


class DiscriminatorLoss:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        raise NotImplementedError

    def get_loss_terms(self):
        return self.terms


class NS_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, D):
        super().__init__([])
        self.D = D
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        ones  = torch.ones_like(real_output, dtype=torch.float, device=device)
        zeros = torch.zeros_like(fake_output, dtype=torch.float, device=device)

        if getattr(self.D, "is_critic", False):
            real_loss = self.bce_logits(real_output, ones)
            fake_loss = self.bce_logits(fake_output, zeros)
        else:
            real_loss = self.bce(real_output, ones)
            fake_loss = self.bce(fake_output, zeros)

        return real_loss + fake_loss, {}


class  W_DiscriminatorLoss(DiscriminatorLoss):
    # (unused in our current construct_loss — we use WGP instead)
    def __init__(self):
        super().__init__([])

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        d_loss_real = -real_output.mean()
        d_loss_fake =  fake_output.mean()
        return d_loss_real + d_loss_fake, {}


class WGP_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, D, lmbda):
        super().__init__(['W_distance', 'D_loss', 'GP'])
        self.D = D
        self.lmbda = lmbda

    def calc_gradient_penalty(self, real_data, fake_data, device):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_data)
        interpolates = (real_data + alpha * (fake_data - real_data)).detach()
        interpolates.requires_grad_(True)

        disc_interpolates = self.D(interpolates)
        grad_outputs = torch.ones_like(disc_interpolates, device=device)

        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]

        gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        return ((gradients_norm - 1.)**2).mean()

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        d_loss_real = -real_output.mean()
        d_loss_fake =  fake_output.mean()
        d_loss = d_loss_real + d_loss_fake

        gp = self.calc_gradient_penalty(real_data, fake_data, device)
        w_distance = - d_loss_real - d_loss_fake

        return d_loss + self.lmbda * gp, {
            'W_distance': w_distance.item(),
            'D_loss':     d_loss.item(),
            'GP':         gp.item()
        }

class HingeR1_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, D, r1_gamma):
        super().__init__(['D_hinge_real', 'D_hinge_fake', 'R1'])
        self.D = D
        self.r1_gamma = r1_gamma

    def calc_r1(self, real_data, device):
        real_data.requires_grad_(True)
        real_logits = self.D(real_data)
        grad_outputs = torch.ones_like(real_logits, device=device)
        grads = autograd.grad(
            outputs=real_logits,
            inputs=real_data,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        # R1 penalty = ||∇ D(x)||^2
        grads = grads.view(grads.size(0), -1)
        return (grads.norm(2, dim=1) ** 2).mean()

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        # 1) Hinge loss
        loss_real = F.relu(1.0 - real_output).mean()
        loss_fake = F.relu(1.0 + fake_output).mean()
        d_loss = 0.5 * (loss_real + loss_fake)

        # 2) R1 penalty
        r1 = self.calc_r1(real_data, device)
        total = d_loss + 0.5 * self.r1_gamma * r1

        return total, {
            'D_hinge_real': loss_real.item(),
            'D_hinge_fake': loss_fake.item(),
            'R1':           r1.item()
        }


class GeneratorLoss:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, device, output):
        raise NotImplementedError

    def get_loss_terms(self):
        return self.terms


class NS_GeneratorLoss(GeneratorLoss):
    def __init__(self, D):
        super().__init__([])
        self.D = D
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()

    def __call__(self, device, output):
        ones = torch.ones_like(output, dtype=torch.float, device=device)
        if getattr(self.D, "is_critic", False):
            return self.bce_logits(output, ones)
        return self.bce(output, ones)


class W_GeneratorLoss(GeneratorLoss):
    def __init__(self):
        super().__init__([])

    def __call__(self, device, output):
        return - output.mean()

class Hinge_GeneratorLoss(GeneratorLoss):
    def __init__(self):
        super().__init__(['G_hinge'])
    def __call__(self, device, fake_output):
        # generator hinge = -E[D(G(z))]
        loss = -fake_output.mean()
        return loss
