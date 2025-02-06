import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal 
from scipy.interpolate import splrep, splev

file_path =r"Buttering_layer_velocity_measurement\TT WBUT LONG 1 WSTR 55DB 100MHZ.capture_acq\data.bin"
filename = os.path.basename(os.path.dirname(file_path))


#Enter parameter values
p= 1 # pitch in mm
dT= 0.01 # dT in micro_seconds
thick=25
reciever_num = 64
cutoff_len = 350
shift_correction = 0

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

sig2=sig[0:500]
sig3=sig[1100:1600]
sig4=np.zeros(600,dtype='<i2')
sig_0=np.concatenate((sig2,sig4,sig3), axis=0)
# sig_0=sig

plt.plot(sig_0[0:len_0],linestyle='-',linewidth=0.5, color='b')
plt.xlabel('Time, data point')
plt.ylabel('Amplitude, A.U.')
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
    
plt.plot(Vel_0)
plt.ylim(5200,6200)
plt.ylabel('Normal beam Velocity, m/s')
plt.xlabel('Element NUmber')   
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show() 


###########################################

transducer_fire = 0

data1=matrix[transducer_fire,:,:]
TOF=[]
VEL=[]
THETA=[]
for i in range (int(reciever_num)):
    
    #len_1=310+abs((transducer_fire-i))*10
    a = 0.25 # Adjust the coefficient to change the curvature of the parabola
    len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
    corr=signal.correlate(sig_0[0:len_1],data1[i][0:len_1])
    max_p=np.argmax(corr)
    tof=(len_1-max_p+tof_0/2)*dT 
    TOF.append(tof)
    dist=(thick**2+((i- transducer_fire)*p)**2)**0.5
    vel=1000*dist/tof
    VEL.append(vel.round(3))
    theta=(np.arctan(((i- transducer_fire)*p)/thick))*180/np.pi
    THETA.append(theta.round(0))


# Plotting the velocity values
plt.figure(figsize=(10, 6))
plt.ylim(5200,6200)
plt.plot(THETA, VEL, linestyle='none', marker='o', color='r', label='Velocity (m/s)')
plt.xlabel('Angle (degree)', fontsize=14)
plt.ylabel('Velocity (m/s)', fontsize=14)
plt.title('Velocity vs. Angle', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


plt.plot(THETA,TOF)





##########################################

#   Cutting B-scan to eliminat the shear wave velocity signal
import numpy as np
import matplotlib.pyplot as plt

# Define variables
transducer_fire =0  
i = np.arange(0, 64)

# Calculate length for each column using a parabolic equation
a = 0.25# Adjust the coefficient to change the curvature of the parabola
len_1 = cutoff_len+ a * (transducer_fire - i)**2


#Initialize BSCAN data with zero padding
max_len = int(np.max(len_1))
BSCAN_plot = np.zeros((max_len, len(i)))

#Fill BSCAN_plot with data 
for idx, length in enumerate(len_1):
    BSCAN_plot[:int(length), idx] = matrix[transducer_fire, :, :].T[:int(length), idx]
    
plt.imshow(BSCAN_plot, aspect='auto', cmap='viridis', vmin = 0, vmax = 100)
plt.colorbar()
plt.title('BSCAN Image with Zero Padding')
plt.xlabel('Transducer')
plt.ylabel('Data Points')
plt.show()


##################################################



# Loop over multiple transducer values
transducer_fires = [i for i in range(64)]  # Example range of transducer indices (0 to 4)
plt.figure(figsize=(12, 8))

for transducer_fire in transducer_fires:
    
    data2=matrix[transducer_fire,:,:]/np.max(matrix[transducer_fire,:,:])
    TOF=[]
    VEL=[]
    THETA=[]
    for i in range (int(reciever_num)):
        
        #len_1=310+abs((transducer_fire-i))*10
        a = 0.25 # Adjust the coefficient to change the curvature of the parabola
        len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
        corr=signal.correlate(sig_0[0:len_1],data2[i][0:len_1])
        max_p=np.argmax(corr)
        tof=(len_1-max_p+tof_0/2)*dT 
        TOF.append(tof)
        dist=(thick**2+((i-transducer_fire+shift_correction)*p)**2)**0.5
        vel=1000*dist/tof
        VEL.append(vel.round(3))
        theta=(np.arctan(((i-transducer_fire)*p)/thick))*180/np.pi
        THETA.append(theta.round(0))
    
    # Plot the velocity values
    plt.plot(THETA, VEL, linestyle='none', marker='o', label=f' {transducer_fire}')

plt.xlabel('Angle (degree)', fontsize=18)
plt.ylabel('Velocity (m/s)', fontsize=18)
plt.title(filename, fontsize=16)
# plt.ylim(5200,6200)
# plt.legend(fontsize=16)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()











# To find the shift in the receiver position
transducer_fires = [3]  
plt.figure(figsize=(12, 8))

for transducer_fire in transducer_fires:
    
    BSCAN=matrix[transducer_fire,:,:]/np.max(matrix[transducer_fire,:,:])
    TOF=[]
    for i in range (int(transducer_fires[0])-3,4+transducer_fires[0]):
        
        #len_1=310+abs((transducer_fire-i))*10
        a = 0.25 # Adjust the coefficient to change the curvature of the parabola
        len_1 = int(400 + a * (transducer_fire - i)**2)
        corr=signal.correlate(sig_0[0:len_1],BSCAN[i][0:len_1])
        # plt.plot(BSCAN[i][0:len_1])
        max_p=np.argmax(corr)
        tof=(len_1-max_p+tof_0/2)*dT 
        TOF.append(tof)
        
DIST1 = []
DIST_ACT = []
for idx,_ in enumerate(np.arange (transducer_fires[0]-3,transducer_fires[0]+4)):

    dist1 = (TOF[idx]*5780/1000)
    DIST1.append(dist1)
    # dist_act = np.sqrt(thick**2+(i-transducer_fires[0])**2)
    # DIST_ACT.append(dist_act)


# Spline fitting DIST1 data points
x = np.arange(transducer_fires[0]-3,transducer_fires[0]+4)  # Index values corresponding to DIST1
spl = splrep(x, DIST1)  # Spline representation of DIST1
parab_coeffs = np.polyfit(x, DIST1, 2)  # Parabolic (quadratic) fit
x_smooth = np.linspace(transducer_fires[0]-3,transducer_fires[0]+3, 100)  # Smooth range for plotting the spline
DIST1_spline = splev(x_smooth, spl)  # Evaluate the spline at the smooth points
DIST1_parabola = np.polyval(parab_coeffs, x_smooth) #parabola fit

# Find the minimum value of the spline fit
min_index = np.argmin(DIST1_spline)
min_x = x_smooth[min_index]
min_y = DIST1_spline[min_index]

# Find the minimum value of the quadratic fit
min_index_p = np.argmin(DIST1_parabola)
min_x_p = x_smooth[min_index_p]
min_y_p = DIST1_parabola[min_index_p]

# Plotting the original DIST1 and DIST_ACT data and the spline fit
plt.plot(x, DIST1, 'o', label="DIST1 (Data Points)", markersize=5)  # Original data points
plt.plot(x_smooth, DIST1_spline, label="DIST1 (Spline Fit)", linestyle='--')  # Spline fit
plt.plot(x_smooth, DIST1_parabola, label="DIST1 (Parabolic Fit)", linestyle='--')  # Parabolic fit
# plt.plot(x, DIST_ACT, label="DIST_ACT", color='green')  # DIST_ACT data
plt.plot(min_x, min_y, 'ro', label=f'Minimum Spline (x={min_x:.2f}, y={min_y:.2f})')
plt.plot(min_x_p, min_y_p, 'ro', label=f'Minimum parabola (x={min_x_p:.2f}, y={min_y_p:.2f})')

print(f'spline min {transducer_fires[0]-min_x}')
print(f'parabola min {transducer_fires[0]-min_x_p}')

# Labels and legends
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Spline Fit for DIST1 and DIST_ACT')
plt.legend()
plt.grid(True)
plt.show()
