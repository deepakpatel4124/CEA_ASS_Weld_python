import numpy as np
import matplotlib.pyplot as plt


# File path to the text file
file_path_receiver = r'E:\Weld FEM Simulation\Python_Comsol\201_Domain_02_09\Weld_Region\small_correct_mesh\T32_201_Domain_improved_mesh.txt'  # Replace with your actual file path
file_path_transducer = r'transducer_fire.txt'  # Replace with your actual file path

# Constants
transmiter_probe = 33
threshold = 0.01 
datapoint_offset = 1000

pitch =1 # mm
thickness = 30 # mm

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
time_transducer = data_transducer[:, 0]
amplitude_transducer = -data_transducer[:, 1]

# Extract time and amplitude data for receivers
time_receiver = data_receiver[:, 0]
amplitude_receivers = data_receiver[:, 1:]

plt.imshow(amplitude_receivers, aspect ='auto')
plt.colorbar()
plt.show()

plt.plot(data_receiver[:, 0],data_receiver[:, 64])
plt.show()


def process_signals(amplitude_receivers, threshold):
    num_rows, num_cols = amplitude_receivers.shape
    processed_signals = np.zeros_like(amplitude_receivers)
    
    for col in range(num_cols):
        signal = amplitude_receivers[:, col]/np.max(amplitude_receivers)
        # Find the first index where the signal crosses the threshold
        crossing_index = np.argmax(signal > threshold)
        
        if signal[crossing_index] > threshold:
            start_index = max(0, crossing_index - 1000)
            end_index = min(num_rows, crossing_index + 1000)
            processed_signals[start_index:end_index, col] = signal[start_index:end_index]
    
    return processed_signals

processed_signals = process_signals(amplitude_receivers, threshold)

# Plotting the processed signals
plt.imshow(processed_signals, aspect='auto')
plt.colorbar()
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
    peak_index_receiver = np.argmax(processed_signals[:,i])
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
# plt.ylim(5200,6200)

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



