import numpy as np
import matplotlib.pyplot as plt

# File paths to the CSV files
csv_file_1 = 'AV_T1_59_Domain_Weld_ortho_Steel_NorVel.txt.csv'  
csv_file_2 = 'AV_T31_59_Domain_Weld_ortho_Steel_NorVel.txt.csv'  
csv_file_3 = 'AV_T64_59_Domain_Weld_ortho_Steel_NorVel.txt.csv'  
csv_file_4 = 'T32_iso_Steel_NorVel.txt.csv' 
csv_file_5 = 'T1_iso_Steel_NorVel.txt.csv' 
csv_file_6 = 'T64_iso_Steel_NorVel.txt.csv' 



# Load the data from the CSV files
data_1 = np.loadtxt(csv_file_1, delimiter=',', skiprows=1)
data_2 = np.loadtxt(csv_file_2, delimiter=',', skiprows=1)
data_3 = np.loadtxt(csv_file_3, delimiter=',', skiprows=1)
# data_4 = np.loadtxt(csv_file_4, delimiter=',', skiprows=1)
# data_5 = np.loadtxt(csv_file_5, delimiter=',', skiprows=1)
# data_6 = np.loadtxt(csv_file_6, delimiter=',', skiprows=1)


# Extract angle and velocity data for both datasets
angle_1 = data_1[:, 0]
velocity_1 = data_1[:, 1]

angle_2 = data_2[:, 0]
velocity_2 = data_2[:, 1]

angle_3 = data_3[:, 0]
velocity_3 = data_3[:, 1]

# angle_4 = data_4[:, 0]
# velocity_4 = data_4[:, 1]

# angle_5 = data_5[:, 0]
# velocity_5 = data_5[:, 1]

# angle_6 = data_6[:, 0]
# velocity_6 = data_6[:, 1]

# Plotting both datasets on the same plot
plt.figure(figsize=(10, 6))
plt.plot(angle_1, velocity_1, marker='.',linestyle='None', color='r', label='AnisoT1', alpha = 0.5)
plt.plot(angle_2, velocity_2, marker='x',linestyle='None', color='b', label='AnisoT32', alpha = 0.5)
plt.plot(angle_3, velocity_3, marker='*',linestyle='None', color='c', label='AnisoT64', alpha = 0.5)
# plt.plot(angle_4, velocity_4, marker='*',linestyle='None', color='b', label='IsoT32', alpha = 0.5)
# plt.plot(angle_5, velocity_5, marker='.',linestyle='None', color='r', label='IsoT1', alpha = 0.5)
# plt.plot(angle_6, velocity_6, marker='x',linestyle='None', color='c', label='IsoT64', alpha = 0.5)


plt.xlabel('Angle (Degree)')
plt.ylabel('Velocity (m/s)')
plt.title('Angle Dependent Ultrasonic Velocity')
plt.grid(True)
plt.legend()
plt.ylim(500,6000)
plt.tight_layout()
plt.show()


plt.plot(angle_1, velocity_1 , color='r', label='Dataset 1', alpha = 0.5)