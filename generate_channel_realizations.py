import numpy as np
import time
import torch
import utils_function
import tensorflow as tf
from typing import Optional
import sys



def generate_noisy_channels_fixed_SNR(channels,snr_in_dB):# channels : shape [nb_channels,64]
    
    Nb_chan=channels.shape[0]
    Nb_BS_antenna=channels.shape[1]
    h_noisy =np.zeros((Nb_chan,Nb_BS_antenna),dtype=np.complex128)
    channels_norm=np.zeros(Nb_chan)
    sigma_2 = np.zeros(Nb_chan) 
    snr_in_lin = 10**(0.1*snr_in_dB)
     
     
    for j in range (Nb_chan):# loop over all samples

        sigma_2[j] = np.linalg.norm(channels[j,:],2)**2/(Nb_BS_antenna*snr_in_lin) 

        #Noise generation
        n = (np.sqrt(sigma_2[j]))*(np.random.randn(Nb_BS_antenna)+1j*np.random.randn(Nb_BS_antenna))
            
        #Add noise to each channel
        h_noisy[j,:] =channels[j,:]+n
        

    channels_norm = np.linalg.norm(h_noisy,2,axis=1) 
    h_noisy = h_noisy/channels_norm[:,None]
    
    
        
   
    return h_noisy,sigma_2      
    
        
        
        
def generate_noisy_channels_varying_SNR(channels,sigma):
    # Add noise to the channels 
           
    Nb_chan=channels.shape[0]
    Nb_BS_antenna=channels.shape[1]
    h_noisy =np.zeros((Nb_chan,Nb_BS_antenna),dtype=np.complex128)
    channels_norm=np.zeros(Nb_chan)
    sigma_2 = np.full(Nb_chan,sigma) 
     
     
    for j in range (Nb_chan):# loop over all samples


        #Noise generation
        n = (np.sqrt(sigma_2[j]))*(np.random.randn(Nb_BS_antenna)+1j*np.random.randn(Nb_BS_antenna))
            
        #Add noise to each channel
        h_noisy[j,:] =channels[j,:]+n
 

    return torch.tensor(h_noisy,dtype=torch.complex128),torch.tensor(sigma_2)        
       
    
    
    