import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal 
from scipy.interpolate import splrep, splev
import os


# Define a list of file paths

# file_paths = [
#     r'Velocity_exp_data\WP#01_TT _velocity_measurements_16_09_2024\Weld pad#1 SN 02 PM ROOT 40DB 1L.capture_acq\data.bin',
#     r'Velocity_exp_data\WP#01_TT _velocity_measurements_16_09_2024\Weld pad#1 SN 03 PM ROOT 40DB 1L.capture_acq\data.bin',
#     r'Velocity_exp_data\WP#01_TT _velocity_measurements_16_09_2024\Weld pad#1 SN 04 PM ROOT 40DB 1L.capture_acq\data.bin',
#     r'Velocity_exp_data\WP#01_TT _velocity_measurements_16_09_2024\Weld pad#1 SN 05 PM ROOT 40DB 1L.capture_acq\data.bin',
#     r'Velocity_exp_data\WP#01_TT _velocity_measurements_16_09_2024\Weld pad#1 SN 06 PM ROOT 40DB 1L.capture_acq\data.bin'
# ]

file_paths = [
             r"E:\Weld FEM Simulation\Weld_exp_data\Velocity_exp_data\TT_Weldpad_#0_#1_22_08_24\WeldPad_#0_SN#0\TT wp 00 no 0 WCL Tr Crown 55 db.capture_acq\data.bin",
             r"E:\Weld FEM Simulation\Weld_exp_data\Velocity_exp_data\TT_Weldpad_#0_#1_22_08_24\WeldPad_#0_SN#0\TT wp 00 no 0 WCL Tr Root 55 db.capture_acq\data.bin",
            #  r"Velocity_exp_data\WP 00 and 01 26 Aug 2024\Weld pad#0 SN 00 PM CROWN 50DB 1L.capture_acq\data.bin",
            #  r"Velocity_exp_data\WP 00 and 01 26 Aug 2024\Weld pad#0 SN 00 PM CROWN 50DB 1L.capture_acq\data.bin"
            ]


#Enter parameter values
p= 1 # pitch in mm
dT= 0.01 # dT in micro_seconds
reciever_num = 64
cutoff_len = 320
vel_const = 5760 ## for receiver shift correction needed


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



# Placeholder for the velocity matrix across files
all_velocities = []

# Loop over each file
for file_path in file_paths:
    
    # Extract the filename from the file path
    filename = os.path.abspath(file_path)
    print(filename)
    # Find the relevant part of the filename (e.g., SN xx WM ROOT)
    start_idx = filename.find('SN')  # Find the index where 'SN' starts
    if start_idx != -1:
        relevant_info = filename[start_idx:start_idx + 18]  # 'SN xx WM ROOT' is 15 characters long
        # Print the extracted information
        print(f"Processing file with: {relevant_info}")
    else:
        print("File name convention not followed")
        
    transducer_num, data_points, file = sub_main(file_path)
    matrix = get_matrix(file, transducer_num, data_points)
    
    # Create a 2x2 grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Processing {relevant_info}', fontsize=16)

    # Plot 1: B-Scan Data for first transducer
    ax1 = axes[0, 0]
    im = ax1.imshow(matrix[0,:,:1600].T, aspect='auto', cmap='viridis', extent=(0, (transducer_num-1)*p, data_points*dT, 0))
    ax1.set_xlabel('Distance, mm')
    ax1.set_ylabel('Time, microsecond')
    ax1.set_title('B-Scan for Transducer 0')
    fig.colorbar(im, ax=ax1)
    
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

    ax2 = axes[0, 1]
    ax2.plot(sig_0[0:len_0], linestyle='-', linewidth=0.5, color='b')
    ax2.set_xlabel('Time, data point')
    ax2.set_ylabel('Amplitude, A.U.')
    ax2.set_title('Signal Correction')
    
    corr=signal.correlate((sig[0:len_0]),(sig[len_0:2*(len_0)]))
    max_p=np.argmax(corr)
    tof_0=2*len_0-max_p  
    print(f"Initial TOF correction = {tof_0}")

    #####################################################################################
    
    ##Shift Correction of the receiver 

    # To find the shift in the receiver position
    
    transducer_fires = [3]  
    

    for transducer_fire in transducer_fires:
        
        BSCAN=matrix[transducer_fire,:,:]/np.max(matrix[transducer_fire,:,:])
        TOF=[]
        for i in range (int(transducer_fires[0]*2+1)):
            a = 0.25 # Adjust the coefficient to change the curvature of the parabola
            len_1 = int(400 + a * (transducer_fire - i)**2)
            corr=signal.correlate(sig_0[0:len_1],BSCAN[i][0:len_1])
            max_p=np.argmax(corr)
            tof=(len_1-max_p+tof_0/2)*dT 
            TOF.append(tof)
            
    DIST_C = []
    for i in np.arange (transducer_fires[0]-3,transducer_fires[0]+4):

        dist_C = (TOF[i]*vel_const/1000)
        DIST_C.append(dist_C)


    # parabola fitting DIST_C data points
    x = np.arange(transducer_fires[0]-3,transducer_fires[0]+4)  # Index values corresponding to DIST_C
    parab_coeffs = np.polyfit(x, DIST_C, 2)  # Parabolic (quadratic) fit
    x_smooth = np.linspace(transducer_fires[0]-3,transducer_fires[0]+3, 100)  # Smooth range for plotting the spline
    DIST_C_parabola = np.polyval(parab_coeffs, x_smooth) #parabola fit


    # Find the minimum value of the quadratic fit
    min_index_p = np.argmin(DIST_C_parabola)
    min_x_p = x_smooth[min_index_p]
    min_y_p = DIST_C_parabola[min_index_p]

    ax3 = axes[1, 0]
    # Plotting the original DIST_C and DIST_ACT data and the spline fit
    ax3.plot(x, DIST_C, 'o', label="DIST_C (Data Points)", markersize=5)  # Original data points
    ax3.plot(x_smooth, DIST_C_parabola, label="DIST_C (Parabolic Fit)", linestyle='--')  # Parabolic fit
    ax3.plot(min_x_p, min_y_p, 'ro', label=f'Minimum parabola (x={min_x_p:.2f}, y={min_y_p:.2f})')

    shift_correction = 3-min_x_p
    thick=min_y_p

    print(f'parabola min {shift_correction}')
    print(f'parabola min {thick}')

    # Labels and legends
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Distance')
    ax3.set_title('Spline Fit for DIST_C and DIST_ACT')
    ax3.legend()
    ax3.grid(True)

    #####################################################################################
    
    
    # Initialize a placeholder for VEL values for this specific file
    file_velocities = []
    ax4 = axes[1, 1]
    plt.figure(figsize=(12, 8))
    # Process transducers for each file
    transducer_fires = [0,32,63]  # You can modify this range as needed
    for transducer_fire in transducer_fires:
        
        data2 = matrix[transducer_fire, :, :] / np.max(matrix[transducer_fire, :, :])
        TOF = []
        VEL = []
        THETA=[]
        for i in range(reciever_num):
            a = 0.25  # Adjust the coefficient to change the curvature of the parabola
            len_1 = int(cutoff_len + a * (transducer_fire - i)**2)
            corr = signal.correlate(sig_0[0:len_1], data2[i][0:len_1])
            max_p = np.argmax(corr)
            tof = (len_1 - max_p + tof_0 / 2) * dT
            TOF.append(tof)
            
            # Calculate the distance and velocity
            dist = (thick**2 + ((i - transducer_fire + shift_correction) * p)**2)**0.5
            vel = 1000 * dist / tof
            VEL.append(vel.round(3))
            theta=(np.arctan(((i-transducer_fire)*p)/thick))*180/np.pi
            THETA.append(theta.round(0))
        file_velocities.append(VEL)  # Append velocity data for the current transducer
        
        # Plot the velocity values
        ax4.plot(THETA, VEL, linestyle='none', marker='o', label=f'Transducer {transducer_fire}')
        plt.plot(THETA, VEL, linestyle='none', marker='o', label=f'Transducer {transducer_fire}')

    plt.xlabel('Angle (degree)', fontsize=18)
    plt.ylabel('Velocity (m/s)', fontsize=18)
    plt.title('Velocity vs. Angle', fontsize=16)
    plt.ylim(5200,6200)
    plt.legend(fontsize=16)
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    all_velocities.append(file_velocities)  # Append the velocities for this file
    
    ax4.set_xlabel('Angle (degree)', fontsize=18)
    ax4.set_ylabel('Velocity (m/s)', fontsize=18)
    ax4.set_title('Velocity vs. Angle', fontsize=16)
    ax4.set_ylim(5200,6200)
    ax4.legend(fontsize=16)
    ax4.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    ax4.minorticks_on()
    ax4.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    ax4.tick_params(axis='x', labelsize=16)  # Set font size for x-axis ticks
    ax4.tick_params(axis='y', labelsize=16) 
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top spacing for the suptitle
    plt.show()
    
##Convert to 2D Matrix and Plot

# Convert the list into a 2D numpy array for better visualization
all_velocities = np.array(all_velocities)

transducer_fire_index = 2  # You can modify this based on which transducer fire you want to visualize

# Extract data for the selected transducer fire
data_to_plot = all_velocities[:, transducer_fire_index, :]  # Shape will be (num_files, num_receivers)

# Plotting the data using imshow
plt.figure(figsize=(12, 8))

plt.imshow(data_to_plot, aspect='auto', cmap='viridis')

# Label the axes
plt.colorbar(label='Velocity (m/s)')
plt.xlabel('Receiver Index', fontsize=18)
plt.ylabel('File Index', fontsize=18)  # Y-axis is for different files (axis 0)
plt.title(f'Velocity 2D Profile for Transducer Fire {transducer_fires[transducer_fire_index]}', fontsize=16)

plt.grid(True)
plt.show()