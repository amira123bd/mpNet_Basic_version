from __future__ import print_function
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.autograd.function import Function, FunctionCtx
import math
import generate_steering

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Custom autograd Functions
class keep_k_max(Function):
    
    @staticmethod
    def forward(ctx: FunctionCtx, activations_in: Tensor, k: int) -> Tuple[Tensor, Tensor]:

        input = activations_in.clone().detach().to(device)
        
        if input.dim() == 1:
            # Wrap single datapoint in tensor of size 2
            input = input.unsqueeze(0)

        n_samples = input.shape[0] 
        d_a = input.shape[1] 

        activations_out = torch.zeros([n_samples, d_a], dtype=torch.complex128, device=device)
        
        input = input.to(torch.complex128)

        id_k_max = torch.zeros([n_samples, k], dtype=torch.int, device=device)

        # Loop over all the examples
        for i in range(n_samples):
            id_k_max[i, :] = torch.topk(torch.abs(input[i, :]), k)[1]
            activations_out[i, id_k_max[i, :]] = input[i, id_k_max[i, :]]

        # Save activations that correspond to the selected atoms for backward propagation
        ctx.save_for_backward(activations_out)
        return activations_out, id_k_max
  
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor, id_k_max: Tensor) -> Tuple[Tensor, None]:
        
        activations_out, = ctx.saved_tensors
       
        grad_input = grad_output.clone().to(device)
        grad_input[activations_out == 0] = 0
        
        return grad_input, None

class mpNet_Constrained(nn.Module):
    def __init__(self, ant_position: torch.float, DoA: Tensor, g_vec: torch.Tensor, lambda_, normalize: bool = True) -> None:
        super().__init__()
        self.ant_position = nn.Parameter(ant_position.to(device))
        self.DoA = DoA.to(device)
        self.g_vec =g_vec.to(device)
        self.normalize = normalize
        self._lambda_ = lambda_

    def forward(self, x: Tensor, k: int, sigma: Optional[float] = None, sc: int = 2) -> Tuple[Tensor, Tensor, Optional[torch.Tensor]]:
        residual = x.clone().to(device)

        
        W = generate_steering.steering_vect_c(self.ant_position, self.DoA, self.g_vec, self._lambda_).to(device).type(torch.complex128)
        
        

        if sigma is None:  # no stopping criterion
            residuals = []
            for iter in range(k):
                z, id_k_max = keep_k_max.apply(residual @ torch.conj(W), 1)
                residual = residual - (z @ W.T)
                residuals.append(residual)

            x_hat = x - residual
            return residual, x_hat, None

        else:  # Use stopping criterion
            N = residual.shape[1]
            if sc == 1:  # SC1
                threshold = pow(sigma, 2) * (N + 2 * math.sqrt(N * math.log(N)))
            elif sc == 2:  # SC2
                threshold = pow(sigma, 2) * (N)
            current_ids = list(range(residual.size(0)))
            depths = torch.zeros(residual.size(0), device=device)
            iter = 0

            while bool(current_ids):
                # Calculate L2 norm of each channel
                res_norm_2 = torch.norm(residual, p=2, dim=1) ** 2

                for i in current_ids[:]:
                    if res_norm_2[i] < threshold[i]:
                        depths[i] = iter
                        current_ids.remove(i)
                    else:
                        z, id_k_max = keep_k_max.apply(residual[i].clone() @ torch.conj(W), 1)
                        residual[i] = residual[i].clone() - (z @ W.T).clone()

                iter += 1
            x_hat = x - residual
            return residual, x_hat, depths

class mpNet(nn.Module):
    def __init__(self, W_init: Tensor) -> None:
        super().__init__()
        self.W = nn.Parameter(W_init.to(device).type(torch.complex128))

    def forward(self, x: Tensor, k: int, sigma: Optional[float] = None, sc: int = 2) -> Tuple[Tensor, Tensor, Optional[torch.Tensor]]:
        residual = x.clone().to(device)

        if sigma is None:  # no stopping criterion
            residuals = []
            for iter in range(k):
                z, id_k_max = keep_k_max.apply(residual @ torch.conj(self.W), 1)
                residual = residual - (z @ self.W.T)
                residuals.append(residual)

            x_hat = x - residual
            return residual, x_hat, None

        else:  # Use stopping criterion
            N = residual.shape[1]
            if sc == 1:  # SC1
                threshold = pow(sigma, 2) * (N + 2 * math.sqrt(N * math.log(N)))
            elif sc == 2:  # SC2
                threshold = pow(sigma, 2) * (N)

            current_ids = list(range(residual.size(0)))
            depths = torch.zeros(residual.size(0), device=device)
            iter = 0

            while bool(current_ids):
                # Calculate L2 norm of each channel
                res_norm_2 = torch.norm(residual, p=2, dim=1) ** 2

                for i in current_ids[:]:
                    if res_norm_2[i] < threshold[i]:
                        depths[i] = iter
                        current_ids.remove(i)
                    else:
                        z, id_k_max = keep_k_max.apply(residual[i].clone() @ torch.conj(self.W), 1)
                        residual[i] = residual[i].clone() - (z @ self.W.T).clone()

                     

                iter += 1
            x_hat = x - residual


            return residual, x_hat, depths
