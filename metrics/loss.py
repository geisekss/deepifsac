import numpy as np
import torch
import torch.nn.functional as F

NCE_TEMP = 0.7
LAM0 = 0.5
LAM1 = 10
LAM2 = 1
LAM3 = 10

def calculate_constrative_loss(model, x_categ_enc, x_cont_enc, x_categ_enc_2, x_cont_enc_2, criterion1):
    aug1 = model.transformer(x_categ_enc, x_cont_enc)
    aug2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
    aug1 = (aug1 / aug1.norm(dim=-1, keepdim=True)).flatten(1, 2)
    aug2 = (aug2 / aug2.norm(dim=-1, keepdim=True)).flatten(1, 2)

    aug1 = model.pt_mlp(aug1)
    aug2 = model.pt_mlp2(aug2)

    logits1 = aug1 @ aug2.t() / NCE_TEMP
    logits2 = aug2 @ aug1.t() / NCE_TEMP
    targets = torch.arange(logits1.size(0)).to(logits1.device)
    loss = LAM0 * (criterion1(logits1, targets) + criterion1(logits2, targets)) / 2
    return loss

def calculate_denoising_loss(model, x_categ, x_cont, x_categ_enc_2, x_cont_enc_2, train_mask_batch, criterion1):
    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
    if con_outs:
        con_outs = torch.cat(con_outs, dim=1)
        # if opt.have_xOrg:
        #     l2 = criterion2(con_outs, x_cont)
        # else:
        #     l2 = F.mse_loss(con_outs * (1 - train_mask_batch), x_cont * (1 - train_mask_batch), reduction='none')
        #     N = (1 - train_mask_batch).sum()
        #     l2 = l2.sum() / N
        #not opt.have_xOrg
        l2 = F.mse_loss(con_outs * (1 - train_mask_batch), x_cont * (1 - train_mask_batch), reduction='none')
        N = (1 - train_mask_batch).sum()
        l2 = l2.sum() / N
    else:
        l2 = 0
    l1 = sum(criterion1(cat_outs[j], x_categ[:, j]) for j in range(1, x_categ.shape[-1]))
    loss = LAM2 * l1 + LAM3 * l2
    return loss

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data):
        
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        
        D_yn = denoise_fn(y + n, sigma)

        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss
    