import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal 

file_path = r'Velocity_exp_data\WP#01_TT _velocity_measurements_16_09_2024\Weld pad#1 SN 02 PM ROOT 40DB 1L.capture_acq\data.bin'

#Enter parameter values
p= 1 # pitch in mm
dT= 0.01 # dT in micro_seconds
thick=17.38
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
len_0 = 500
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

plt.plot(Vel_0)
plt.xlabel('Angle (degree)', fontsize=14)
plt.ylabel('Velocity (m/s)', fontsize=14)
plt.title('Velocity vs. Angle', fontsize=16)
# plt.ylim(5200,6200)

plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
#########################################################################


#   Cutting B-scan to eliminat the shear wave velocity signal

# Define variables
transducer_fire =0  
i = np.arange(0, 64)


# # Calculate length for each column using a parabolic equation

len_1 = cutoff_len + a * (transducer_fire - i)**2


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



##############################################################################

### vel v/s Theta for multiple transducer

# Loop over multiple transducer values
transducer_fires = [0,31,63]  # Example range of transducer indices (0 to 4)
plt.figure(figsize=(12, 8))

# Initialize vmin and vmax with None
vmin = None
vmax = None


for transducer_fire in transducer_fires:
    
    BSCAN=matrix[transducer_fire,:,:]/np.max(matrix[transducer_fire,:,:])
    TOF=[]
    VEL=[]
    THETA=[]
    for i in range (int(reciever_num)):
        
        #len_1=310+abs((transducer_fire-i))*10
         # Adjust the coefficient to change the curvature of the parabola
        len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
        corr=signal.correlate(sig_0[0:len_1],BSCAN[i][0:len_1])
        max_p=np.argmax(corr)
        tof=(len_1-max_p+tof_0/2)*dT 
        TOF.append(tof)
        dist=(thick**2+((i-transducer_fire)*p)**2)**0.5
        vel=1000*dist/tof
        VEL.append(vel.round(3))
        theta=(np.arctan(((i-transducer_fire)*p)/thick))*180/np.pi
        THETA.append(theta.round(0))
    
    # Update vmin and vmax
    current_min = min(VEL)
    current_max = max(VEL)
    
    if vmin is None or current_min < vmin:
        vmin = current_min
        
    if vmax is None or current_max > vmax:
        vmax = current_max
        
    # Plot the velocity values
    plt.plot(THETA, VEL, linestyle='none', marker='o', label=f'Transducer {transducer_fire}')


plt.xlabel('Angle (degree)', fontsize=14)
plt.ylabel('Velocity (m/s)', fontsize=14)
plt.title('Velocity vs. Angle', fontsize=16)
# plt.ylim(5200,6200)

plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

############################################################################################

####            POLAR REPRESENTATION           ################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import signal

# Your existing code to compute TOF, VEL, THETA, etc.
transducer_fires = [0,31,63]  # Multiple transducer_fire values
scale_factor = 25
# vmin = 5200
# vmax = 6200

if "root" in file_path.lower():
    pos_y = 0
    rotate = 90
    flip = -1
else:
    pos_y = 30
    rotate = 270
    flip = 1

# Load the data from the text file
weld_curve_data = np.loadtxt('weld_coordinates.txt', delimiter=' ', skiprows=1)
x_weld = weld_curve_data[:, 0]
y_weld = weld_curve_data[:, 1]

# Ensure no duplicate x values (optional step depending on your data)
x_weld_unique, unique_indices = np.unique(x_weld, return_index=True)
y_weld_unique = y_weld[unique_indices]

# Cubic Spline Interpolation
cubic_spline_interp = CubicSpline(x_weld_unique, y_weld_unique)
x_new = np.linspace(x_weld.min(), x_weld.max(), num=500)
y_cubic = cubic_spline_interp(x_new)

# Create the base plot
fig, ax = plt.subplots(figsize=(18, 6))

if "wcl" in file_path.lower():
    # Plot the base data (weld curve)
    ax.plot(x_new, y_cubic, '-', linewidth=2, c = 'r')

else:
    # Plot the base data (weld curve)
    ax.plot()

# Loop over multiple transducer_fire values
for transducer_fire in transducer_fires:
    BSCAN = matrix[transducer_fire, :, :] / np.max(matrix[transducer_fire, :, :])
    TOF = []
    VEL = []
    THETA = []
    
    for i in range(int(reciever_num)):
          # Adjust the coefficient to change the curvature of the parabola
        len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
        corr = signal.correlate(sig_0[0:len_1], BSCAN[i][0:len_1])
        max_p = np.argmax(corr)
        tof = (len_1 - max_p + tof_0 / 2) * dT
        TOF.append(tof)
        dist = (thick**2 + ((i - transducer_fire) * p)**2)**0.5
        vel = 1000 * dist / tof
        VEL.append(vel.round(3))
        theta = (np.arctan(((i - transducer_fire) * p) / thick)) * 180 / np.pi
        THETA.append(theta)

    
    # Define the arrow starting coordinates
    pos_x = transducer_fire + 60
    
    rotated_theta = [flip*theta + rotate for theta in THETA]

    # Apply color mapping for arrows
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    for theta, vel in zip(rotated_theta, VEL):
        theta_rad = np.deg2rad(theta)
        end_x = pos_x + scale_factor * (vel / vmax) * np.cos(theta_rad)
        end_y = pos_y + scale_factor * (vel / vmin) * np.sin(theta_rad)
        
        # Determine color based on velocity
        color = cmap(norm(vel))
        
        # Plot the arrow
        ax.arrow(pos_x, pos_y, end_x - pos_x, end_y - pos_y,
                 head_width=0.1, head_length=0.1, fc=color, ec=color, width=0.02)

# Add the color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Velocity Magnitude')


# Customize plot appearance
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
# plt.xlim(20, 160)
# plt.ylim(0, 30)
ax.grid(True)
ax.legend()
ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.show()






##############################################################################################################

transducer_fires = [0,31,63]  # Multiple transducer_fire values
scale_factor = 75
# vmin = 5200
# vmax = 6200

if "root" in file_path.lower():
    pos_y = 0
    rotate = 90
    flip = -1
else:
    pos_y = 30
    rotate = 270
    flip = 1

# Ensure no duplicate x values (optional step depending on your data)
x_weld_unique, unique_indices = np.unique(x_weld, return_index=True)
y_weld_unique = y_weld[unique_indices]

# Cubic Spline Interpolation
cubic_spline_interp = CubicSpline(x_weld_unique, y_weld_unique)
x_new = np.linspace(x_weld.min(), x_weld.max(), num=500)
y_cubic = cubic_spline_interp(x_new)


# Create subplots
num_plots = len(transducer_fires)
fig, axes = plt.subplots(1, num_plots, figsize=(12, 1.5), sharex=True, sharey=True)

# Loop over multiple transducer_fire values and corresponding axes
for idx, transducer_fire in enumerate(transducer_fires):
    ax = axes[idx]
    
    BSCAN = matrix[transducer_fire, :, :] / np.max(matrix[transducer_fire, :, :])
    TOF = []
    VEL = []
    THETA = []
    
    for i in range(int(reciever_num)):
          # Adjust the coefficient to change the curvature of the parabola
        len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
        corr = signal.correlate(sig_0[0:len_1], BSCAN[i][0:len_1])
        max_p = np.argmax(corr)
        tof = (len_1 - max_p + tof_0 / 2) * dT
        TOF.append(tof)
        dist = (thick**2 + ((i - transducer_fire) * p)**2)**0.5
        vel = 1000 * dist / tof
        VEL.append(vel.round(3))
        theta = (np.arctan(((i - transducer_fire) * p) / thick)) * 180 / np.pi
        THETA.append(theta)

    
    # Define the arrow starting coordinates
    pos_x = transducer_fire + 60

    
    rotated_theta = [(flip*theta + rotate) for theta in THETA]

    # Apply color mapping for arrows
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    for theta, vel in zip(rotated_theta, VEL):
        theta_rad = np.deg2rad(theta)
        end_x = pos_x + vel * np.cos(theta_rad)
        end_y = pos_y + vel * np.sin(theta_rad)
        
        # Determine color based on velocity
        color = cmap(norm(vel))
        
        # Plot the arrow
        ax.arrow(pos_x, pos_y, end_x - pos_x, end_y - pos_y,
                 head_width=0.1, head_length=0.1, fc=color, ec=color, width=0.02)

    if "wcl" in file_path.lower():
        # Plot the base data (weld curve)
        ax.plot(x_new, y_cubic, '-', linewidth=2, c = 'r')

    else:
        # Plot the base data (weld curve)
        ax.plot()
        
    ax.set_title(f'Transducer Fire = {transducer_fire}')
    ax.grid(True)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.xlim(59, 125)
    plt.ylim(0, 30)
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')

# Add the color bar to the last subplot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('Velocity Magnitude')
axes[0].set_xlabel('X axis')
axes[0].set_ylabel('Y axis')
plt.xlim(59, 125)
plt.ylim(0, 30)

# Show the plot
plt.show()


###############################################################

transducer_fires = [0, 4, 8, 12, 16, 20, 24, 30,31, 32, 44, 48, 52, 56, 60, 63]
scale_factor = 75


# Settings based on file_path conditions
if "root" in file_path.lower():
    pos_y = 0
    rotate = 90
    flip = -1
else:
    pos_y = 30
    rotate = 270
    flip = 1

# Ensure no duplicate x values (optional step depending on your data)
x_weld_unique, unique_indices = np.unique(x_weld, return_index=True)
y_weld_unique = y_weld[unique_indices]

# Cubic Spline Interpolation
cubic_spline_interp = CubicSpline(x_weld_unique, y_weld_unique)
x_new = np.linspace(x_weld.min(), x_weld.max(), num=500)
y_cubic = cubic_spline_interp(x_new)

# Create subplots in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten to iterate easily

# Loop over multiple transducer_fire values and corresponding axes
for idx, transducer_fire in enumerate(transducer_fires):
    ax = axes[idx]
    
    BSCAN = matrix[transducer_fire, :, :] / np.max(matrix[transducer_fire, :, :])
    TOF = []
    VEL = []
    THETA = []
    
    for i in range(int(reciever_num)):
          # Adjust the coefficient to change the curvature of the parabola
        len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
        corr = signal.correlate(sig_0[0:len_1], BSCAN[i][0:len_1])
        max_p = np.argmax(corr)
        tof = (len_1 - max_p + tof_0 / 2) * dT
        TOF.append(tof)
        dist = (thick**2 + ((i - transducer_fire) * p)**2)**0.5
        vel = 1000 * dist / tof
        VEL.append(vel.round(3))
        theta = (np.arctan(((i - transducer_fire) * p) / thick)) * 180 / np.pi
        THETA.append(theta)

    # Define the arrow starting coordinates
    pos_x = transducer_fire + 60
    rotated_theta = [(flip * theta + rotate) for theta in THETA]

    # Apply color mapping for arrows
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    for theta, vel in zip(rotated_theta, VEL):
        theta_rad = np.deg2rad(theta)
        end_x = pos_x + vel * np.cos(theta_rad)
        end_y = pos_y + vel * np.sin(theta_rad)
        
        # Determine color based on velocity
        color = cmap(norm(vel))
        
        # Plot the arrow
        ax.arrow(pos_x, pos_y, end_x - pos_x, end_y - pos_y,
                 head_width=0.1, head_length=0.1, fc=color, ec=color, width=0.02)

    if "wcl" in file_path.lower():
        # Plot the base data (weld curve)
        ax.plot(x_new, y_cubic, '-', linewidth=2, c = 'r')

    else:
        # Plot the base data (weld curve)
        ax.plot()
            
    ax.set_title(f'Transducer Fire = {transducer_fire}')
    ax.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    plt.xlim(59, 125)
    plt.ylim(0, 30)
# If fewer subplots are used, remove the empty subplot(s)
for idx in range(len(transducer_fires), len(axes)):
    fig.delaxes(axes[idx])

# Add the color bar to the last subplot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes, orientation='vertical')
cbar.set_label('Velocity Magnitude')
plt.xlim(59, 125)
plt.ylim(0, 30)
# Show the plot
plt.show()
