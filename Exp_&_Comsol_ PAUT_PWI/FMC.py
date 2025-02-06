import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate


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

file_path = r'D:\Py_Codes\Python FMC-TFM\Weld_exp_data\1 40 db.capture_acq\data.bin'
pitch = 0.9 # in mm
sampling_frequency = 100e6  # in Hz
thickness = 30 #mm
total_receiver_probe = 64
total_transmitter_probe = 64
gap = 0.1 #mm


transducer_num, data_points, file = sub_main(file_path)
matrix = get_matrix(file, transducer_num, data_points)

#B-Scan Plot
plt.imshow(np.transpose(matrix[0]), aspect = 'auto')
plt.gca().invert_yaxis()
plt.show()

y = matrix[0][0]  # Second column (Pressure (Pa), Boundary Probe 1)
x = np.linspace(0,(len(y)/sampling_frequency),len(y))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x,y, linestyle='-',linewidth=1, color='b', label=f'Pressure (Pa), Boundary Probe {0}')
plt.plot(y, linestyle='-',linewidth=1, color='b', label=f'Pressure (Pa), Boundary Probe {0}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (A.U)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#Velocity Calculation

ref_sig = y[240:300]
sig0 = y[:1000]
sig1 = y[1000:2000]

# plt.plot(ref_sig, linestyle='-',linewidth=1, color='b')

# Compute the cross-correlation
peak_diff = np.argmax(np.correlate(sig0, sig1, mode='full'))

velocity = (2*thickness/1000)/(peak_diff/sampling_frequency)

correlation_shift = np.argmax(correlate(matrix[0][0][:1200], ref_sig, mode='valid'))+(len(ref_sig)/2)

data_shift = (((thickness/1000)/ velocity )*sampling_frequency)-correlation_shift





# transducer_probe = 0
# receiver_probe = 50

# data = matrix[transducer_probe][15]
# plt.plot(matrix[transducer_probe][receiver_probe][:1200], linestyle='-',linewidth=1, color='b')
# plt.show()

# # Find the location of the reference signal in sig1
# correlation = np.argmax(correlate(matrix[transducer_probe][receiver_probe][:1200], ref_sig, mode='valid'))+(len(ref_sig)/2)


transducer_probe = 30  

time_delay = []  # List to store correlation values

# Loop through each receiver probe from 0 to 64
for receiver_probe in range(total_receiver_probe):
    # Extract the signal for the current receiver probe
    sig = matrix[transducer_probe][receiver_probe][:1500]

    # Compute the correlation and find the peak location
    correlation_peak = np.argmax(correlate(sig, ref_sig, mode='valid')) + int(len(ref_sig) / 2)
    print(correlation_peak)
    time = (correlation_peak+data_shift)/sampling_frequency
    # Store the correlation peak in the list
    time_delay.append(time)


#Distance Calculation

# Calculate the x-coordinates of points in the upper and lower rows
x_coords_upper = np.linspace(0, (total_receiver_probe - 1) * (pitch + gap), total_receiver_probe)
x_coords_lower = np.linspace(0, (total_receiver_probe - 1) * (pitch + gap), total_receiver_probe)

# y-coordinates
y_upper = 0
y_lower = thickness

# Calculate angles
angles = np.arctan2(x_coords_lower - x_coords_upper[transducer_probe], y_lower - y_upper)

# Convert angles to degrees for easier interpretation
angles_degrees = np.degrees(angles)

# Calculate distances
distances = np.sqrt((x_coords_lower - x_coords_upper[transducer_probe])**2 + (y_lower - y_upper)**2)

# Store angles and distances in a numpy array
angle_dist_array = np.vstack((angles_degrees, distances)).T

print(angle_dist_array[:,1])


velocity = (angle_dist_array[:,1]/1000)/time_delay


# Plotting the velocity values
plt.figure(figsize=(10, 6))
plt.plot(angle_dist_array[:,0], velocity, linestyle='-', marker='o', color='r', label='Velocity (m/s)')
plt.xlabel('Angle(degree)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs. Receiver Probe')
plt.legend()
plt.grid(True)
plt.show()

















# Plot the reference signal
plt.plot(ref_sig, linestyle='-', linewidth=1, color='b', label='Reference Signal')
plt.legend()
plt.show()

# Find the location of the reference signal in sig0
auto_correlation_sig0 = np.argmax(correlate(sig0, ref_sig, mode='valid'))
print(f"Reference signal peak in sig0 located at index: {auto_correlation_sig0}")

# Find the location of the reference signal in sig1
auto_correlation_sig1 = np.argmax(correlate(sig1, ref_sig, mode='valid'))
print(f"Reference signal peak in sig1 located at index: {auto_correlation_sig1}")

# Calculate the peak difference in samples
peak_diff = auto_correlation_sig1+1000 - auto_correlation_sig0

# Calculate the velocity
velocity = (2 * thickness / 1000) / (peak_diff / sampling_frequency)

print(f"Calculated velocity: {velocity} m/s")



import numpy as np

from scipy import signal


correlation = signal.correlate(sig0, sig1, mode="full")

lags = signal.correlation_lags(sig0.size, sig1.size, mode="full")

lag = lags[np.argmax(correlation)]

plt.plot(correlation)