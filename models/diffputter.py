import os 
import numpy as np
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics.loss import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]

randn_like=torch.randn_like

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) 
        x = x + emb
        return self.mlp(x)


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp = False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):

        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()

class DiffPuter(nn.Module):

    def __init__(self, 
                 in_dim: int,
                 result_save_path: str = "../", 
                 num_trials: int = 10, 
                 epochs_m_step: int = 10000, 
                 patience_m_step: int = 300, 
                 hid_dim: int = 1024, 
                 device: str = "cuda", 
                 lr: float = 1e-4, 
                 num_steps: int = 50, 
                 ckpt_dir: str = "ckpt"):
        super().__init__()
        self.in_dim = in_dim
        # parameters for the whole training step
        self.ckpt_dir = ckpt_dir
        self.result_save_path = result_save_path

        # parameters for M step
        self.epochs_m_step = epochs_m_step
        self.patience_m_step = patience_m_step
        self.hid_dim = hid_dim
        self.lr = lr
        self.device = device
        
        # parameters for E step
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.SIGMA_MIN=0.002
        self.SIGMA_MAX=80
        self.rho=7
        self.S_churn= 1
        self.S_min=0
        self.S_max=float('inf')
        self.S_noise=1
        denoise_fn = MLPDiffusion(self.in_dim, self.hid_dim).to(self.device)
        self.model = Model(denoise_fn=denoise_fn, hid_dim=self.in_dim).to(self.device)
        


    def fit(self, iteration, train_loader):
        ## M-Step: Density Estimation
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=40)
        self.model.train()

        best_loss = float('inf')
        patience = 0
        for epoch in range(self.epochs_m_step):

            batch_loss = 0.0
            len_input = 0
            for batch in train_loader:
                inputs = batch.float().to(self.device)
                loss = self.model(inputs)

                loss = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(self.model.state_dict(), f'{self.ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == self.patience_m_step:
                    print('Early stopping')
                    break
            print(f'Epoch {epoch+1}/{self.epochs_m_step}, Loss: {curr_loss:.4f}, Best loss: {best_loss:.4f}')
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), f'{self.ckpt_dir}/{iteration}/model_{epoch}.pt')


    def transform(self, iteration, X, mask, impute_X):
        rec_Xs = []

        for trial in range(self.num_trials):
            # ==========================================================
            print(f'Trial = {trial}')
            net = self.model.denoise_fn_D

            num_samples, dim = X.shape[0], X.shape[1]

            impute_X = torch.tensor(impute_X).to(self.device)
            mask = mask.to(torch.int).to(self.device)
            rec_X = self.impute_mask(net, impute_X, mask, num_samples, dim)
            
            mask_float = mask.to(torch.float).to(self.device)
            rec_X = rec_X * mask_float + impute_X * (1-mask_float)
            rec_Xs.append(rec_X)
            
        rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 
        rec_X = rec_X.cpu().numpy() * 2 

    
        np.save(f'{self.ckpt_dir}/iter_{iteration+1}.npy', rec_X)
        return rec_X

        
    def compute_metrics(self, iteration: int, rec_Xs: torch.tensor, X_true: torch.tensor, mask: torch.tensor) -> Tuple[float, float]:
        ## E-Step: Missing Value Imputation
        # reconstructed X is the mean value of each rec_X made in each trial of the E-step
        rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 

        # multiplication by 2 due to standardization. MinMaxScaler should enter here
        rec_X = rec_X.cpu().numpy() * 2 
        X_true = X_true.cpu().numpy() * 2 

        np.save(f'{self.ckpt_dir}/iter_{iteration+1}.npy', rec_X)
        
        pred_X = rec_X 
        
        mae, rmse= self.get_eval(pred_X, X_true, mask)

        return mae, rmse

    
    def impute_mask(self, net, x, mask, num_samples, dim):
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=self.device)
        x_t = torch.randn([num_samples, dim], device=self.device)

        sigma_min = max(self.SIGMA_MIN, net.sigma_min)
        sigma_max = min(self.SIGMA_MAX, net.sigma_max)

        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (
                    sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_t = x_t.to(torch.float32) * t_steps[0]

        N = 10
        with torch.no_grad():

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                if i < self.num_steps - 1:
            
                    for j in range(N):
                        n_curr = torch.randn_like(x_t).to(self.device) * t_cur
                        n_prev = torch.randn_like(x_t).to(self.device) * t_next

                        x_known_t_prev = x + n_prev
                        x_unknown_t_prev = self.sample_step(net, i, t_cur, t_next, x_t)

                        x_t_prev = x_known_t_prev * (1-mask) + x_unknown_t_prev * mask

                        n = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                        if j == N - 1:
                            x_t = x_t_prev                                                # turn to x_{t-1}
                        else:
                            x_t = x_t_prev + n                                            # new x_t

        return x_t

    def sample_step(self, net, i, t_cur, t_next, x_next):

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur) 
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * randn_like(x_cur)
        # Euler step.

        denoised = net(x_hat, t_hat).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < self.num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
    
    def get_eval(self, X_recon: np.array, X_true: np.array, mask: np.array) -> Tuple[float, float]:
        X_true = X_true.cpu().numpy() * 2
        mask = mask.cpu().numpy()
        mask = mask.astype(bool)

        mae = float(np.nanmean(np.abs(X_recon[mask]- X_true[mask])))
        rmse = float(np.sqrt(np.nanmean((X_recon[mask]- X_true[mask])**2)))

        return mae, rmse