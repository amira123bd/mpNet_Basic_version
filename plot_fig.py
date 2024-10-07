import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
   
   
path_init = Path.cwd()



def plot_SNR(self,nb_chan):
       
 
        plt.rcParams['text.usetex'] = True
       
       
        plt.figure()
        vec_base = np.arange(0,self.batch_size/self.batch_subsampling,1)
        plt.plot(vec_base,self.snr_out_mpnet,'x-',color='red',linewidth=0.7,label='mpNet ')
        #plt.plot(vec_base,self.snr_out_c_mpnet,'>-',color='red',linewidth=0.7,label='mpNet Constrained ')
        plt.plot(vec_base,self.snr_out_mp_nominal,'*-',color='c',linewidth=0.7,label='MP (Nominal dictionnary) ')
        plt.plot(vec_base,self.snr_out_mp_real,'+-',color='k',linewidth=0.7,label='MP (Real dictionnary)')
        #plt.plot(vec_base,self.snr_out_mp_real,'+-',color='k',linewidth=0.7,label='MP (Real dictionnary)')
        plt.ylabel('SNR_out (dB)')
        plt.xlabel(f'Number of seen channels (*{self.batch_subsampling*nb_chan})')
        plt.legend(loc = 'lower right')
        plt.title(f'SNR_out evolution ')
        plt.grid()
        plt.xlim(left=0)
       
        plt.savefig(path_init/'Results'/('SNR_out_SNR_in_' + \
           str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
               str(nb_chan)+'_channels_per_batch_'),dpi=500)
        plt.show()
 
 
def plot_NMSE_Online(self,nb_chan):
        plt.rcParams['text.usetex'] = True
        plt.figure()
        vec_base = np.arange(0,self.batch_size/self.batch_subsampling,1)
        plt.plot(vec_base,self.NMSE_mpnet,'p-',linewidth=0.8,label='mpNet')  
        #plt.plot(vec_base,self.cost_func_test_c,'>-',linewidth=0.8,label='mpNet Constrained ')
        plt.plot(vec_base,self.NMSE_mp_real,'*-',linewidth=0.8,label='MP (Real dictionnary) ')
        plt.plot(vec_base,self.NMSE_mp_nominal,'+-',linewidth=0.8,label='MP (Nominal dictionnary) ')  
        plt.plot(vec_base,self.NMSE_ls,'o--',linewidth=0.8,label='LS')  
        plt.grid()
        plt.legend(loc = 'best')
        plt.xlabel(f'Number of seen channels (*{self.batch_subsampling*nb_chan})')
        plt.ylabel('NMSE')
        plt.title(f'NMSE evolution ')
        plt.xlim(left=0)
 
        plt.savefig(path_init/'Results'/('NMSE_SNR_in_' + \
            str(self.epochs)+'epoch_'+str(self.batch_size)+'_batch_'+\
                str(nb_chan)+'_channels_per_batch_'),dpi=500)
       
        plt.show()