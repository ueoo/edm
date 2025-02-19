import os
import pickle
import re

import click
import numpy as np
import PIL.Image
import torch
import tqdm

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import dnnlib

from torch_utils import distributed as dist


num_steps = 18.0
sigma_min = 0.002
sigma_max_700 = 700.0
sigma_max_80 = 80.0
rho = 7.0
S_churn = 0.0
S_min = 0.0
S_max = float("inf")
S_noise = 1.0

print("EDM sampler args")
print(f"num_steps: {num_steps}")
print(f"sigma_min: {sigma_min}")
print(f"sigma_max_700: {sigma_max_700}")
print(f"sigma_max_80: {sigma_max_80}")
print(f"rho: {rho}")
print(f"S_churn: {S_churn}")
print(f"S_min: {S_min}")
print(f"S_max: {S_max}")
print(f"S_noise: {S_noise}")

# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
t_steps_700 = (
    sigma_max_700 ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max_700 ** (1 / rho))
) ** rho
t_steps_700 = torch.cat([torch.as_tensor(t_steps_700), torch.zeros_like(t_steps_700[:1])])  # t_N = 0

t_steps_80 = (
    sigma_max_80 ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max_80 ** (1 / rho))
) ** rho
t_steps_80 = torch.cat([torch.as_tensor(t_steps_80), torch.zeros_like(t_steps_80[:1])])  # t_N = 0

indices = torch.cat([step_indices, torch.zeros_like(step_indices[:1]) + num_steps])

os.makedirs("vis", exist_ok=True)
plt.plot(indices.cpu().numpy(), t_steps_700.cpu().numpy(), "o-")
plt.plot(indices.cpu().numpy(), t_steps_80.cpu().numpy(), "o-")
plt.xlabel("Step")
plt.ylabel("t")
plt.legend([f"sigma_max={sigma_max_700}", f"sigma_max={sigma_max_80}"])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
out_name = f"vis/t_steps_steps{num_steps}_sigmin{sigma_min}_rho{rho}.png"
plt.savefig(out_name, dpi=300)
plt.close()
