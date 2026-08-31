import numpy as np
import torch
import torch.nn as nn


def get_entropy_coefficient(config, env, device):
    compile_mode = config.algorithm.compile_mode
    entropy_coefficient = EntropyCoefficient(config, env, device).to(device)
    if not bool(config.algorithm.compile_policy):
        return entropy_coefficient
    entropy_coefficient = torch.compile(entropy_coefficient, mode=compile_mode)
    entropy_coefficient.forward = torch.compile(entropy_coefficient.forward, mode=compile_mode)
    entropy_coefficient.loss = torch.compile(entropy_coefficient.loss, mode=compile_mode)
    return entropy_coefficient


class EntropyCoefficient(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.target_entropy = config.algorithm.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -torch.prod(torch.tensor(np.prod(env.single_action_space.shape), dtype=torch.float32).to(device)).item()
        else:
            self.target_entropy = float(self.target_entropy)
        alpha_init = float(getattr(config.algorithm, "alpha_init", 1.0))
        if alpha_init <= 0.0:
            raise ValueError("algorithm.alpha_init must be positive")
        self.log_alpha = nn.Parameter(
            torch.full((1,), np.log(alpha_init), device=device)
        )
    
    
    def forward(self):
        return self.log_alpha.exp()
    

    def loss(self, entropy):
        return self.log_alpha.exp() * (entropy - self.target_entropy)
