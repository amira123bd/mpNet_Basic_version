#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
#get real and nominal antenna pos
path_init=Path.cwd()
file_name = 'Data2/datasionna/antenna_position.npz'  
antenna_pos = np.load(path_init/file_name)
nominal_ant_positions=antenna_pos['nominal_position']
real_ant_positions=antenna_pos['real_position']
shift=0.03
BS_position=[-302, 42, 23.0]
lambda_ = 0.010706874

# %%
plt.rcParams['text.usetex'] = True

plt.figure(figsize=(10, 6))

plt.figure(figsize=(8, 6), dpi=100)  # Adjust size and DPI as needed
plt.scatter(nominal_ant_positions[3:8][:,1], np.zeros_like(nominal_ant_positions[0:5][:,0]), label='Nominal positions', marker='x')


plt.scatter((real_ant_positions[3:8][:,1]), np.zeros_like(real_ant_positions[:5][:,0]), label='Real positions', marker='x')

plt.xlabel('Y coordinate (m)', fontsize=10, labelpad=12)
plt.ylabel('', fontsize=10, labelpad=12)


plt.title(r'Nominal and Real antennas positions with uncertainty of 0.1 $ \lambda$')
plt.legend()

plt.grid(True)
plt.show()



# %%
plt.savefig(os.path.join(path_init, 'variance_0.03.png'), dpi=500)
