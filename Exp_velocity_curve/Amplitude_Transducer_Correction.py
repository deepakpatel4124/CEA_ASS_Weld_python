import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal 

file_path = r'Velocity_exp_data\WP 00 and 01 26 Aug 2024\tt wp-00 sn 00 50db TR  WCL  TRANS.capture_acq\data.bin'

#Enter parameter values
p= 1 # pitch in mm
dT= 0.01 # dT in micro_seconds
thick=28.96
reciever_num = 64
cutoff_len = 200
a = 0.25 #adjust the parabola curvature

def sub_main(file_path):
    file = np.fromfile(file_path, dtype="<i2", sep='')
    transducer_num = int(file[55])
    data_points = int(file[63])
    return transducer_num, data_points, file

def get_matrix(file, transducer_num, data_points):
    file = file[64:]
    matrix = np.array([])
    for i in range(transducer_num):
        start = (data_points + 10) * i * transducer_num + i * 10
        end = start + (data_points + 10) * transducer_num
        asc = file[start:end]
        d = np.array([])
        d = np.append(d, asc)
        d = d.reshape(transducer_num, data_points + 10)
        matrix = np.append(matrix, d[:, :data_points])

    matrix = matrix.reshape(transducer_num, transducer_num, data_points)
    return matrix

transducer_num, data_points, file = sub_main(file_path)
matrix = get_matrix(file, transducer_num, data_points)


plt.xlabel('Distance, mm')
plt.ylabel('Time, microsecond')
plt.imshow(matrix[0,:,:1600].T, aspect='auto', cmap = 'viridis', extent=(0, (transducer_num-1)*p, data_points*dT, 0) )
plt.show()

#### initial correction for transducer firing using tranducer = 0

transducer_fire = 0
BSCAN_ref=matrix[transducer_fire,:,:]

k = 0  # Receiver Number
len_0 = 800
sig=(BSCAN_ref[k]/(np.max(BSCAN_ref[k])))
sig1 = sig[0:1600]
# sig2=sig[0:500]
# sig3=sig[1100:1600]
# sig4=np.zeros(600,dtype='<i2')
# sig_0=np.concatenate((sig2,sig4,sig3), axis=0)
sig_0=sig

# Create the 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(sig, linestyle='-', linewidth=0.5, color='b')
axs[0, 0].set_title('sig')

axs[0, 1].plot(sig[0:len_0], linestyle='-', linewidth=0.5, color='b')
axs[0, 1].set_title(f'sig[0:{len_0}]')
axs[0, 1].set_xticks([]) 
axs[1, 0].plot(sig[len_0:2*(len_0)], linestyle='-', linewidth=0.5, color='b')
axs[1, 0].set_title(f'sig[{len_0}:{2*len_0}]')
axs[1, 0].set_xticks([]) 
axs[1, 1].plot(sig_0[0:len_0], linestyle='-', linewidth=0.5, color='b')
axs[1, 1].set_title(f'sig_0[0:{len_0}]')
axs[1, 1].set_xticks([]) 
plt.tight_layout()
plt.show()

corr=signal.correlate((sig[0:len_0]),(sig[len_0:2*(len_0)]))
max_p=np.argmax(corr)
tof_0=2*len_0-max_p  
print(f"shift = {tof_0}")

##############################################

# Below is to calculate the Normal Velocity using firing and receiving from same transducer

Theta=[]
Vel=[]
TOF_1=[]
Vel_0=[]

for k in range(int(reciever_num)):
    
    sig_N=(matrix[k,:,:][k]/(np.max(matrix[k,:,:][k])))
    corr=signal.correlate((sig_0[0:len_0]), (sig_N[0:len_0]))
    max_p=np.argmax(corr)
    tof_1=(len_0-max_p+tof_0/2)*dT
    vel=1000*thick/tof_1 #m/s
    TOF_1.append(tof_1)
    Vel_0.append(vel)

# Normal velocoty plot
plt.plot(Vel_0)
plt.xlabel('Transducer No.', fontsize=14)
plt.ylabel('Velocity (m/s)', fontsize=14)
plt.title('Velocity vs. Angle', fontsize=16)
plt.ylim(5200,6200)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()





#########################################################################

####   Below is to correct the amplitude of the transducer fire #######


import numpy as np
import matplotlib.pyplot as plt

max_amp = []

for k in range(int(reciever_num)):
    max_amp.append(np.max(matrix[k,:,:][k]))

# Calculate the mean peak amplitude across all transducer-receiver pairs
mean_amp = np.mean(max_amp/np.max(max_amp))

# Calculate the correction factors
correction_factors = mean_amp / np.array(max_amp)

# Apply the correction factors to the signals
corrected_matrix = np.empty_like(matrix)
for k in range(int(reciever_num)):
    corrected_matrix[k,:,:] = matrix[k,:,:] * correction_factors[k]

# Calculate corrected peak amplitudes
corrected_max_amp = [np.max(corrected_matrix[k,:,:][k]) for k in range(int(reciever_num))]

# Plot original and corrected peak amplitudes
plt.figure(figsize=(12, 6))

# Plot original peak amplitudes
plt.plot(max_amp/np.max(max_amp), label='Original Amplitudes', color='blue')

# Plot corrected peak amplitudes
plt.plot(corrected_max_amp, label='Corrected Amplitudes', color='red')

# Add labels and grid
plt.xlabel('Transducer No.', fontsize=14)
plt.ylabel('Amplitude[A.U]', fontsize=14)
plt.ylim(0,1)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)

# Show the plot
plt.show()
