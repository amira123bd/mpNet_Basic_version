import numpy as np
import sys
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import torch




# generate steering vectors
def steering_vect_c(antenna_pos : np.ndarray, DoA : np.ndarray, g_vec : np.ndarray, lambda_ : float ) \
    -> Tuple[np.ndarray]:
    
        N=np.size(antenna_pos,0) #Number of antenna in the BS
        A=np.size(DoA,0) #Number of Atoms
        
        dict_=torch.zeros((N,A),dtype=torch.complex128)
        
    
                    
                    
        expo=torch.exp(-1j * 2 * np.pi * (1 / lambda_) * torch.matmul(antenna_pos, DoA.T))
                    
        for j in range(N):
            dict_[j,:]=g_vec[j]*expo[j]
            
        #print(type(dict_))
        return dict_/torch.tensor(np.sqrt(np.sum(np.abs(dict_.detach().numpy())**2,0)))
    
    
    
    
    