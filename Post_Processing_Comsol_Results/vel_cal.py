import numpy as np
import matplotlib.pyplot as plt


# File path to the text file
file_path_receiver = r'201_Domain_02_09\Weld_Region\small_correct_mesh\T32_201_Domain_improved_mesh_27mm_thick_crown.txt'  # Replace with your actual file path
file_path_transducer = 'transducer_fire.txt'  # Replace with your actual file path

# Constants
transmiter_probe = 31
cutoff_len = 1200    
a = 3# Adjust the coefficient to change the curvature of the parabola

total_receiver_probe = 64
pitch =1 # mm
thickness = 27 # mm

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

# Read the data
data_receiver = read_data(file_path_receiver)
data_transducer = read_data(file_path_transducer)

# Extract time and amplitude data for transducer
time_transducer = data_transducer[:10000, 0]
amplitude_transducer = -data_transducer[:10000, 1]

# Extract time and amplitude data for receivers
time_receiver = data_receiver[:10000, 0]
amplitude_receivers = data_receiver[:10000, 1:]

plt.imshow(amplitude_receivers, aspect ='auto')
plt.colorbar()
plt.show()

plt.plot(data_receiver[:, 0],data_receiver[:, 64])
plt.show()

###This is to cut unwanted signal)#####
i = np.arange(0, 64)
len_1 = np.argmax(amplitude_receivers[:,transmiter_probe])+cutoff_len + a * (transmiter_probe - i)**2
BSCAN_plot = np.zeros((amplitude_receivers.shape))
#Fill BSCAN_plot with data 
for idx, length in enumerate(len_1):
    BSCAN_plot[:int(length), idx] = amplitude_receivers[:int(length), idx]
    
plt.imshow(BSCAN_plot, aspect='auto', vmax = 1e-9)
plt.colorbar()
plt.title('BSCAN Image with Zero Padding')
plt.xlabel('Transducer')
plt.ylabel('Data Points')
plt.show()

# Identify the peak in the transducer data
peak_index_transducer = np.argmax(amplitude_transducer)
peak_time_transducer = time_transducer[peak_index_transducer]

# Initialize array to store time delays
time_delays = np.zeros(amplitude_receivers.shape[1])
peak_amplitude_receiver = np.zeros(amplitude_receivers.shape[1])
TOF=[]
VEL=[]
THETA=[]

# Calculate time delays for each receiver probe
for i in range(amplitude_receivers.shape[1]):
    len_1 = np.argmax(amplitude_receivers[:,transmiter_probe])+cutoff_len + a * (transmiter_probe - i)**2
    peak_index_receiver = np.argmax((amplitude_receivers[:int(len_1), i]))
    peak_time_receiver = time_receiver[peak_index_receiver]
    time_delays[i] = np.abs(peak_time_receiver - peak_time_transducer)
    peak_amplitude_receiver[i] = np.max((amplitude_receivers[:, i]))
    dist=(thickness**2+((i-transmiter_probe)*pitch)**2)**0.5
    print(dist)
    vel=dist/(1000*time_delays[i])
    VEL.append(vel)
    theta=(np.arctan2(((i-transmiter_probe)*pitch), thickness))*180/np.pi
    THETA.append(theta)

# Plotting time delay
plt.plot(THETA,time_delays, marker = '.', color='r', label=f'Pressure (Pa), Transmiter Probe {transmiter_probe}')

plt.xlabel('Theta')
plt.ylabel('Time delay')
plt.show()

# Plotting velocity
plt.figure(figsize=(10, 6))
plt.plot(THETA,VEL, marker = '.', color='r', label=f'Pressure (Pa), Transmiter Probe {transmiter_probe}')
plt.xlabel('Angle(Degree)')
plt.ylabel('Velocity (m/s)')
plt.ylim(5200,6200)

plt.title('Angle Dependent Ultrasonic Velocity')
plt.legend()
plt.grid(True, which='major', linestyle='-', linewidth=0.75, color='black')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

plt.plot(peak_amplitude_receiver/(np.max(peak_amplitude_receiver)))
plt.xlabel('Transducer(No.)')
plt.ylabel('Peak Amplitude (A.U))')
plt.show()

plt.plot(-data_transducer)



