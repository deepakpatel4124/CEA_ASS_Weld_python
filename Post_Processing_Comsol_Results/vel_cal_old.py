import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Constants
pitch = 1  # mm
thickness = 30  # mm
a = 3  # Adjust the coefficient to change the curvature of the parabola
cutoff_len = 10000


# Function to parse and clean the text file
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the starting line of the data (skipping the header)
    start_line = 0
    for i, line in enumerate(lines):
        if not line.startswith('%'):
            start_line = i
            break
    
    # Extract the data lines
    data_lines = lines[start_line:]

    # Clean the data
    cleaned_data = []
    for line in data_lines:
        # Remove extra spaces and split by spaces
        split_line = line.split()
        if len(split_line) == 0:  # Skip empty lines
            continue
        # Convert to float if possible, otherwise skip the line
        try:
            float_values = [float(value) for value in split_line]
            cleaned_data.append(float_values)
        except ValueError:
            continue
    
    # Convert cleaned data to numpy array
    data = np.array(cleaned_data)
    return data

file_path_transducer = 'transducer_fire.txt'  # Replace with your actual file path
data_transducer = read_data(file_path_transducer)
time_transducer = data_transducer[:, 0]
amplitude_transducer = -data_transducer[:, 1]

# Function to process data and plot velocity vs angle
def process_and_plot_velocity(file_path_receiver, transducer_probe):
    # Read the receiver data
    data_receiver = read_data(file_path_receiver)

    # Extract time and amplitude data for receivers
    time_receiver = data_receiver[:, 0]
    amplitude_receivers = data_receiver[:, 1:]

    # Identify the peak in the transducer data
    peak_index_transducer = np.argmax(amplitude_transducer)
    peak_time_transducer = time_transducer[peak_index_transducer]

    # Initialize array to store time delays
    time_delays = np.zeros(amplitude_receivers.shape[1])
    peak_amplitude_receiver = np.zeros(amplitude_receivers.shape[1])
    TOF = []
    VEL = []
    THETA = []

    # Calculate time delays for each receiver probe
    for i in range(amplitude_receivers.shape[1]):
        len_1 = np.argmax(amplitude_receivers[:, transducer_probe]) + cutoff_len + a * (transducer_probe - i) ** 2
        peak_index_receiver = np.argmax(amplitude_receivers[:int(len_1), i])
        peak_time_receiver = time_receiver[peak_index_receiver]
        time_delays[i] = np.abs(peak_time_receiver - peak_time_transducer)
        peak_amplitude_receiver[i] = np.max(amplitude_receivers[:, i])
        dist = (thickness ** 2 + ((i - transducer_probe) * pitch) ** 2) ** 0.5
        vel = dist / (1000 * time_delays[i])
        VEL.append(vel)
        theta = (np.arctan2(((i - transducer_probe) * pitch), thickness)) * 180 / np.pi
        THETA.append(theta)

    # Plotting velocity
    plt.plot(THETA, VEL, marker='.', linestyle = 'none', label=f'Transducer Fire: {transducer_probe}')
    plt.xlabel('Angle (Degree)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Angle Dependent Ultrasonic Velocity')
    plt.ylim(5200,6200)
    plt.grid(True)
    
    
    
    
    
    
    
    
    
    
    
    
def plot_comparison(file_paths, transducer_probes):
    plt.figure(figsize=(10, 6))
    for file_path, transducer_probe in zip(file_paths, transducer_probes):
        process_and_plot_velocity(file_path, transducer_probe)

    plt.ylim(5450,6050)

    plt.legend(fontsize=16)
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


    plt.tight_layout()
    plt.show()

# Define file paths
file_paths = [
    r'201_Domain_02_09\Weld_Region\Large_incorrect_mesh\T1_Root_201Domain.txt', 
    r'201_Domain_02_09\Weld_Region\Large_incorrect_mesh\T32_Root_201Domain.txt', 
    r'201_Domain_02_09\Weld_Region\Large_incorrect_mesh\T64_Root_201Domain.txt'
]
transducer_probes = [0,32,63]  # Define the corresponding transducer probe numbers for each file

plot_comparison(file_paths, transducer_probes)















##########################  Normal Velocity #########################################################


def normal_velocity(file_paths, thickness, cutoff_len):
    VEL = []
    transducer_numbers = []

    for file_path_receiver in file_paths:
        # Read the receiver data
        data_receiver = read_data(file_path_receiver)
        
        # Extract the transducer fire number
        file_name = os.path.basename(file_path_receiver)
        match = re.match(r'^T(\d+)', file_name)
        if match:
            transducer_probe = int(match.group(1))
            transducer_numbers.append(transducer_probe)
        
        # Extract time and amplitude data for receivers
        time_receiver = data_receiver[:, 0]

        # Identify the peak in the receiver data
        len_1 = np.argmax(data_receiver[:, transducer_probe]) + cutoff_len
        peak_index_receiver = np.argmax(data_receiver[:int(len_1), transducer_probe])
        peak_time_receiver = time_receiver[peak_index_receiver]

        peak_index_transducer = np.argmax(amplitude_transducer)
        peak_time_transducer = time_transducer[peak_index_transducer]

        # Calculate velocity
        time_delays = np.abs(peak_time_receiver - peak_time_transducer)
        vel = thickness / (1000 * time_delays)
        VEL.append(vel)

    # Plot the velocities for each transducer
    plt.figure(figsize=(10, 6))
    plt.ylim(5200,6200)
    plt.plot(transducer_numbers, VEL, marker='o', linestyle='-', color='blue')
    plt.xlabel('Transducer Number', fontsize=14)
    plt.ylabel('Normal Velocity (m/s)', fontsize=14)
    plt.title('Normal Velocity vs. Transducer Number', fontsize=16)
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')  
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return VEL

# Process and plot velocities
VEL = normal_velocity(file_paths, thickness, cutoff_len)


















#################  Amplitude correction #####################


# Define file paths
file_paths = [
    r'201_Domain_26_08\Parent_Material\T1_201_Domain_PM.txt', 
    r'201_Domain_26_08\Parent_Material\T15_201_Domain_PM.txt', 
    r'201_Domain_26_08\Parent_Material\T32_201_Domain_PM.txt',
    r'201_Domain_26_08\Parent_Material\T34_201_Domain_PM.txt',
    r'201_Domain_26_08\Parent_Material\T50_201_Domain_PM.txt',
    r'201_Domain_26_08\Parent_Material\T64_201_Domain_PM.txt', 
]


def amp_transducer_fire(file_paths):
    max_amp = []
    transducer_numbers = []
    
    for file_path_receiver in file_paths:
        data_receiver = read_data(file_path_receiver)
        
        # Extract the transducer fire number
        file_name = os.path.basename(file_path_receiver)
        match = re.match(r'^T(\d+)', file_name)
        if match:
            transducer_probe = int(match.group(1))
            transducer_numbers.append(transducer_probe)
            
        max_amp.append(np.max(data_receiver[:, transducer_probe]))
    print(transducer_numbers)
    # Plot the max amplitudes for all files
    plt.figure(figsize=(10, 6))
    plt.ylim(-0.2,1.2)
    plt.plot(transducer_numbers, max_amp/np.max(max_amp), marker='o', linestyle='-', color='blue')
    plt.xlabel('FTransducer No.', fontsize=14)
    plt.ylabel('Amplitude [A.U.]', fontsize=14)
    plt.title('Amplitude for Different Transducer Fire', fontsize=16)
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')  
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


# Call the function with the list of file paths
amp_transducer_fire(file_paths)





################################################################################################################

