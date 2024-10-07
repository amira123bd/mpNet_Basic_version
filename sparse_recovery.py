from typing import Tuple
import numpy as np
import sys
import torch

def omp(x: np.ndarray, D: np.ndarray,
        n_it_max: int = 1000, verbose: bool = True, tol: float = 0.0) -> Tuple[np.ndarray, np.array, np.ndarray]:
    r"""Orthogonal matching pursuit (OMP) algorithm

    Greedy sparse recovery algorithm, minimizing residual error.
    List of most correlated steering vectors considering the input noisy channel.
    This results in a denoised output channel.`

    Args:
        x (np.ndarray): input (noisy channel), :math:`x`
        E (np.ndarray): dictionary of steering vectors, :math:`\mathbf{E}`
        n_it_max (int, optional): maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): display residual norm per iteration. Defaults to True.
        tol (int, optional): tolerance margin of residual error. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.array, np.ndarray]: Tuple composed of:
                                                 - the input estimate (denoised channel), :math:`\hat{\mathbf{h}}`
                                                 - the coefficients of each most correlated steering vector, :math:`\mathbf{e}_s^H . \mathbf{r}`
                                                 - array of most correlated atom index per iteration
    """
    residual = x  # residual error, Algo1.step1
    # array of indices of most correlated steering vectors
    #I = np.empty(0, dtype=int)
    n_it = 0
    stop = False
    while stop == False:  # Algo1.step2
        # Finding which index to add to the support
        a = torch.conj(D.T) @ residual / torch.norm(D, p=2, dim=0)  # Algo1.step3
        idx = torch.abs(a).argmax()  # index of most correlated atom
        # Augmenting the temporary dictionary with the chosen index
        if n_it == 0:
            D_I = D[:, idx]
            D_I = D_I.reshape((D_I.size()[0], 1))
            #I = np.concatenate((I, np.array([idx])))
        else:
            D_idx = D[:, idx]
            D_idx = D_idx.reshape((D_idx.size()[0], 1))
            D_I = torch.hstack((D_I, D_idx))
           # I = np.concatenate((I, np.array([idx])))
   
        # Determining the value of the coefficients
        # difference with MP; orthogonality ensured this way
        coeffs = torch.linalg.pinv(D_I) @ x
        # Computing the new residual
        h_hat = D_I @ coeffs
        residual = x - h_hat  # Algo1.step4
        if verbose:
            print('Iter ' + str(n_it+1) + ', Residual norm = ' +
                  str(torch.norm(residual, p=2)))
        n_it += 1
        stop = torch.norm(residual,2)**2 < tol 
    return h_hat, coeffs



def mp(x: torch.Tensor, E: torch.Tensor,
       n_iter_max: int = 1000, verbose: bool = True, tol: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Matching pursuit (MP) algorithm using PyTorch

    Greedy sparse recovery algorithm, minimizing residual error.
    List of most correlated steering vectors considering the input noisy channel.
    This results in a denoised output channel.
    See :math:`\mathbf{Algorithm 1}`

    Args:
        x (torch.Tensor): input (noisy channel), :math:`x`
        E (torch.Tensor): dictionary of steering vectors, :math:`\mathbf{E}`
        n_iter_max (int, optional): maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): display residual norm per iteration. Defaults to True.
        tol (float, optional): tolerance margin of residual error. Defaults to 0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple composed of:
                                                 - the input estimate (denoised channel), :math:`\hat{\mathbf{h}}`
                                                 - the coefficients of each most correlated steering vector, :math:`\mathbf{e}_s^H . \mathbf{r}`
                                                 - tensor of most correlated atom index per iteration
    """
    residual = x.clone()  # residual error, Algo1.step1
    I = torch.empty(0, dtype=torch.long)  # tensor of indices of most correlated steering vectors
    n_it = 0
    stop = False
    
    while not stop:  # Algo1.step2
        # Finding which index to add to the support
        a = torch.matmul(torch.conj(E.T), residual)   # Algo1.step3
        idx = torch.argmax(torch.abs(a))  # index of most correlated atom
        
        # Augmenting the temporary dictionary with the chosen index
        if n_it == 0:
            E_I = E[:, idx].unsqueeze(1)
            I = torch.cat((I, torch.tensor([idx], dtype=torch.long)))
            coeffs = a[idx].unsqueeze(0)
        else:
            E_idx = E[:, idx].unsqueeze(1)
            E_I = torch.cat((E_I, E_idx), dim=1)
            I = torch.cat((I, torch.tensor([idx], dtype=torch.long)))
            coeffs = torch.cat((coeffs, a[idx].unsqueeze(0)))
        
        h_hat = torch.matmul(E_I, coeffs)
        residual = x - h_hat  # Algo1.step4
        
        if verbose:
            print(f'Iter {n_it+1}, Residual norm = {torch.norm(residual).item()}')
        
        n_it += 1
        stop = torch.norm(residual, 2)**2 < tol 
    
    return h_hat, coeffs