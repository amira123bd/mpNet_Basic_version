import matplotlib.pyplot as plt
from typing import Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import generate_steering
from torch import autograd
import sparse_recovery
import sys
import mpnet_model
import time



# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class UnfoldingModel_Sim:
    def __init__(self,
                 nominal_antenna_pos: np.ndarray,
                 real_antenna_pos: np.ndarray,
                 DoA: np.ndarray,
                 g_vec: np.ndarray,
                 lambda_,             
                 lr: float = 0.001,
                 lr_constrained: float =0.1,
                 momentum: float = 0.9,                 
                 optimizer = 'adam',
                 epochs: int = 10,
                 batch_size: int = 100,               
                 k: int = None, 
                 batch_subsampling: int = 20,
                 train_type: str = 'online',
                 f0:int=28e9,
                 ) -> None:


        self.lr=lr
        self.lr_constrained=lr_constrained
        self.momentum=momentum
        self.epochs=epochs
        self.k=k
        self.batch_subsampling=batch_subsampling
        self.train_type=train_type
        self.nominal_antenna_pos=torch.tensor(nominal_antenna_pos).type(torch.FloatTensor).to(device)
        self.DoA=torch.tensor(DoA).type(torch.FloatTensor).to(device)
        self.g_vec=torch.tensor(g_vec).to(device)
        self.lambda_= lambda_
        self.train_type=train_type
        self.f0=f0
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.real_antenna_pos=torch.tensor(real_antenna_pos).type(torch.FloatTensor).to(device)
        self.times = None
        self.start_time = None

        
        # to store the channel realizations
        self.dataset=None
      
        #sigma noise
        self.sigma_noise = None
        
        #stopping criteria
        self.SC2 = None


        #generate nominal steering vectors using nominal antenna positions
        dict_nominal=generate_steering.steering_vect_c(self.nominal_antenna_pos,
                                                       self.DoA,
                                                       self.g_vec,
                                                       self.lambda_).to(device)
        
    

        dict_real = generate_steering.steering_vect_c(self.real_antenna_pos,
                                                       self.DoA,
                                                       self.g_vec,
                                                       self.lambda_).to(device)
        
        

        self.dict_nominal = dict_nominal
        self.dict_real=dict_real
        weight_matrix=dict_nominal
        
        #initialization of the mpnet model with the weight matrix
        self.mpNet = mpnet_model.mpNet(weight_matrix).to(device)
        self.mpNet_Constrained= mpnet_model.mpNet_Constrained(self.nominal_antenna_pos,self.DoA,self.g_vec,self.lambda_, True).to(device)
        
        #Initialization of the optimizer
        self.optimizer= optim.Adam(self.mpNet.parameters(),lr=self.lr)
        self.constrained_optimizer = optim.Adam(self.mpNet_Constrained.parameters(), lr = self.lr_constrained)

        
        #Result table for every batch size over the whole epochs 
        self.cost_func = np.zeros(epochs*batch_size)
        self.cost_func_c=np.zeros(epochs*batch_size)
        
        
        # Results each batch subsampling
        dim_result=int(np.ceil(epochs*batch_size/batch_subsampling))

        #Initialization of SNR out table 
        self.snr_out_c_mpnet=np.zeros(dim_result)
        self.snr_out_mpnet=np.zeros(dim_result)
        self.snr_out_mp_real=np.zeros(dim_result)
        self.snr_out_mp_nominal=np.zeros(dim_result)
        self.snr_out_ls = np.zeros(dim_result)
        
        #Initialization of NMSE table
        self.NMSE_mpnet= np.zeros(dim_result)
        self.NMSE_c_mpnet = np.zeros(dim_result)
        self.NMSE_mp_real =np.zeros(dim_result)
        self.NMSE_mp_nominal = np.zeros(dim_result)
        self.NMSE_ls = np.zeros(dim_result)


        self.times=np.zeros(dim_result)
        self.cost_func_position=np.zeros(dim_result)
        
        
        
    def train_online_test_inference(self):


        # Load test data
        path_init = Path.cwd()
        file_name = 'Data2/data_var_snr/test_data.npz'
        test_data = np.load(path_init / file_name)

        h_clean_test = torch.tensor(test_data['h'], dtype=torch.complex128).to(device)  # clean channels
        h_noisy_test = torch.tensor(test_data['h_noisy'], dtype=torch.complex128).to(device)  # noisy channels
        sigma_2_test = torch.tensor(test_data['sigma_2']).to(device)  # Noise variance
        norm_test = torch.norm(h_noisy_test, p=2, dim=1).to(device)  # channel's norm

        sigma_norm_test = (torch.sqrt(sigma_2_test) / norm_test).to(device)
        h_clean_test = h_clean_test / norm_test[:, None]
        h_noisy_test = h_noisy_test / norm_test[:, None]

        SC2 = pow(sigma_norm_test, 2) * h_clean_test.shape[1]

     

        idx_subs = 1
        for batch in range(self.batch_size):
            if batch == 0:
                self.start_time=time.time()
                with torch.no_grad():
                    residuals_mp_test, est_chan_test, test_temp_test = self.mpNet(h_noisy_test, self.k, sigma_norm_test, 2)
                    residuals_mp_c_test, est_chan_c_test, test_c_temp_test = self.mpNet_Constrained(h_noisy_test, self.k, sigma_norm_test, 2)


                h_hat_mpNet_test = est_chan_test.detach().cpu().numpy()
                self.NMSE_mpnet[0] = torch.mean(torch.sum(torch.abs(h_clean_test.cpu() - h_hat_mpNet_test) ** 2, 1) / torch.sum(torch.abs(h_clean_test.cpu()) ** 2, 1))

                h_hat_mpNet_c_test = est_chan_c_test.detach().cpu().numpy()
                self.NMSE_c_mpnet[0] = torch.mean(torch.sum(torch.abs(h_clean_test.cpu() - h_hat_mpNet_c_test) ** 2, 1) / torch.sum(torch.abs(h_clean_test.cpu()) ** 2, 1))

                self.snr_out_mpnet[0] = 10 * np.log10(1 / self.NMSE_mpnet[0])

                snr_out_mp_real, NMSE_mp_real = UnfoldingModel_Sim.run_mp_omp(self, h_clean_test, h_noisy_test, 'MP', SC2, 'real')
                snr_out_mp_nominal, NMSE_mp_nominal = UnfoldingModel_Sim.run_mp_omp(self, h_clean_test, h_noisy_test, 'MP', SC2, 'nominal')

                self.NMSE_mp_real[:] = NMSE_mp_real * np.ones(len(self.NMSE_mp_real))
                self.snr_out_mp_real[:] = snr_out_mp_real * np.ones(len(self.snr_out_mp_real))
                self.NMSE_mp_nominal[:] = NMSE_mp_nominal * np.ones(len(self.NMSE_mp_nominal))
                self.snr_out_mp_nominal[:] = snr_out_mp_nominal * np.ones(len(self.snr_out_mp_nominal))

                cost_func_ls = np.mean(np.linalg.norm(h_clean_test.cpu() - h_noisy_test.cpu(), 2, axis=1) ** 2 / np.linalg.norm(h_clean_test.cpu(), 2, axis=1) ** 2)
                self.NMSE_ls[:] = cost_func_ls * np.ones(len(self.NMSE_ls))
                
                self.snr_out_ls[:] = (10 * np.log10(1 / self.NMSE_ls[0]) ) * np.ones(len(self.snr_out_mp_real))


                self.cost_func_position[0]= np.linalg.norm(self.nominal_antenna_pos.cpu()-self.real_antenna_pos.cpu(),2)**2/np.linalg.norm(self.real_antenna_pos.cpu(),2)**2



                print(f'NMSE batch {batch} MpNet: {self.NMSE_mpnet[0]}')
                print(f'NMSE batch {batch} MP Nominal: {self.NMSE_mp_nominal[0]} ')
                print(f'NMSE batch {batch} MP Real: {self.NMSE_mp_real[0]} ')
                print(f'NMSE batch {batch} LS: {self.NMSE_ls[0]}')

            # Load Train channel
            file_name = f'Data2/data_var_snr/batch_{batch}.npz'
            train_data = np.load(path_init / file_name)
            h_clean_train = torch.tensor(train_data['h'], dtype=torch.complex128).to(device)
            h_noisy_train = torch.tensor(train_data['h_noisy'], dtype=torch.complex128).to(device)
            sigma_2_train = torch.tensor(train_data['sigma_2']).to(device)

            norm_train = torch.norm(h_noisy_train, p=2, dim=1).to(device)

            h_clean_train = h_clean_train / norm_train[:, None]
            h_noisy_train = h_noisy_train / norm_train[:, None]

            self.sigma_noise = (torch.sqrt(sigma_2_train) / norm_train).to(device)

            self.optimizer.zero_grad()
            self.constrained_optimizer.zero_grad()
            residuals_mp, est_chan, test_temp = self.mpNet(h_noisy_train, self.k, self.sigma_noise, 2)
            residuals_mp_c, est_chan_c, test_temp_c = self.mpNet_Constrained(h_noisy_train, self.k, self.sigma_noise, 2)

            out_mp = torch.abs(residuals_mp).pow(2).sum() / h_clean_train.size()[0]
            out_mp.backward()
            self.optimizer.step()

            out_mp_c = torch.abs(residuals_mp_c).pow(2).sum() / h_clean_train.size()[0]
            out_mp_c.backward()
            self.constrained_optimizer.step()



            if batch % self.batch_subsampling == 0 and batch != 0:
                self.times[idx_subs] = time.time() - self.start_time
                with torch.no_grad():
                    residuals_mp_test, est_chan_test, test_temp_test = self.mpNet(h_noisy_test, self.k, sigma_norm_test, 2)
                    residuals_mp_c_test, est_chan_c_test, test_c_temp_test = self.mpNet_Constrained(h_noisy_test, self.k, sigma_norm_test, 2)

                h_hat_mpNet_test = est_chan_test.detach().cpu().numpy()
                self.NMSE_mpnet[idx_subs] = torch.mean(torch.sum(torch.abs(h_clean_test.cpu() - h_hat_mpNet_test) ** 2, 1) / torch.sum(torch.abs(h_clean_test.cpu()) ** 2, 1))
                self.snr_out_mpnet[idx_subs] = 10 * np.log10(1 / self.NMSE_mpnet[idx_subs])

                h_hat_mpNet_c_test = est_chan_c_test.detach().cpu().numpy()
                self.NMSE_c_mpnet[idx_subs] = torch.mean(torch.sum(torch.abs(h_clean_test.cpu() - h_hat_mpNet_c_test) ** 2, 1) / torch.sum(torch.abs(h_clean_test.cpu()) ** 2, 1))



                for name, param in self.mpNet_Constrained.named_parameters():
                    
                        estim_position=param.detach().cpu().numpy() 
                        
                self.cost_func_position[idx_subs]= np.linalg.norm(estim_position-self.real_antenna_pos.cpu().numpy(),2)**2/np.linalg.norm(self.real_antenna_pos.cpu().numpy(),2)**2



                print(f'NMSE batch {batch} MpNet: {self.NMSE_mpnet[idx_subs]}')
                print(f'NMSE batch {batch} MP Nominal: {self.NMSE_mp_nominal[idx_subs]} ')
                print(f'NMSE batch {batch} MP Real: {self.NMSE_mp_real[idx_subs]} ')

                idx_subs += 1


            
            
    def run_mp_omp(self,clean_channels : np.ndarray, noisy_channels : np.ndarray,
                   mode : str, SC: np.ndarray, type_dict : str)->Tuple[float,float]:
        
        out_chans = torch.zeros_like(noisy_channels,dtype=torch.complex128)

        for i in range(noisy_channels.shape[0]):
            if type_dict == 'nominal':
                if mode == 'MP':
                    out_chans[i,:], coeffs = sparse_recovery.mp(noisy_channels[i,:],self.dict_nominal,self.k,False,SC[i])
                elif mode == 'OMP':
                    out_chans[i,:], coeffs= sparse_recovery.omp(noisy_channels[i,:],self.dict_nominal,self.k,False,SC[i])
                else: 
                    sys.exit('Unknown algo for MP/OMP')
            elif type_dict == 'real':
                if mode == 'MP':
                    out_chans[i,:], coeffs = sparse_recovery.mp(noisy_channels[i,:],self.dict_real,self.k,False,SC[i])
                elif mode == 'OMP':
                    out_chans[i,:], coeffs = sparse_recovery.omp(noisy_channels[i,:],self.dict_real,self.k,False,SC[i])
                else: 
                    sys.exit('Unknown algo for MP/OMP')
            else:
                sys.exit('Unknown type')

        NMSE = torch.mean(torch.sum(torch.abs(out_chans.cpu()-clean_channels.cpu())**2,1)/torch.sum(torch.abs(clean_channels.cpu())**2,1))
        # snr_out_dB = 10*np.log10(np.mean(np.linalg.norm(clean_channels,2,axis=1)**2)/(np.mean(np.linalg.norm(clean_channels-out_chans,2,axis=1)**2)))
        mean_snr_out_dB = 10*np.log10(1/NMSE)

        

    
        return mean_snr_out_dB, NMSE
                 
        
        
        
                        
       


              

    
     
       
        
        

       
                    
   
            
            
            
            
            
            
            
        
        
        