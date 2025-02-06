import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig
import itertools
from numba import njit, prange


# List of file paths (add all your file paths here)
file_paths = [
    r'PWI_data\PWI_Simulation\PWI_PM_Defect_MM_WC\PWI_PM_Defect_MM_WC_theta_60.txt'
    # r'F:\PWI_PM_Correct\data_correct\PM_Defect_1_theta_70.txt',
    # r'F:\PWI_PM_Correct\data_correct\PM_Defect_1_theta_80.txt'   
]

AngStart = 60
AngEnd = 61
AngStep = 1

# Parameter definitions
p = 1.0  # pitch in mm
N = 64
vel = 5.780  # velocity in mm/us
thickness = 28.5
PW = (N - 1) * p  # PW is the probe width
# Define custom colormap
color_list = [
    (0.0, "white"),  # white
    (0.33, "blue"),  # blue
    (0.67, "yellow"),  # yellow
    (1.0, "red"),  # red
]
cmap = colors.LinearSegmentedColormap.from_list("mycmap", color_list)

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

# Function to extract the theta value from the filename
def extract_theta_from_filename(filename):
    # Assuming the filename is in the format: "PM_Defect_{number}_theta_{theta_value}.txt"
    basename = os.path.basename(filename)  # Get only the file name, not the path
    theta_part = basename.split('_')[-1]   # Extract the last part which contains theta value and .txt
    theta_value = theta_part.replace('.txt', '')  # Remove '.txt' from theta part
    return float(theta_value)

# Main function to read all files and combine them
def process_files(file_paths):
    combined_data = []
    theta_values = []
    time_receiver = None
    
    for file_path in file_paths:
        # Extract the theta value from the filename
        theta_value = extract_theta_from_filename(file_path)
        
        # Read the data from the file
        data = read_data(file_path)
        
        # Extract time column
        current_time_receiver = data[:, 0]  # First column is time
        
        # Check if time_receiver is the same for all files
        if time_receiver is None:
            # Store the first file's time as reference
            time_receiver = current_time_receiver
        else:
            # Compare time values with the reference
            if not np.array_equal(time_receiver, current_time_receiver):
                raise ValueError(f"Time values in {file_path} do not match the first file!")
        
        # Store the theta value
        theta_values.append(theta_value)
        
        # Remove the first two columns (time and theta should not be included)
        amplitude_data = data[:, 1:]  # Assuming the first column is time and you want the rest

        # Ensure it has exactly 64 receivers (columns) per file
        if amplitude_data.shape[1] != 64:
            raise ValueError(f"Expected 64 receivers, but got {amplitude_data.shape[1]} in file {file_path}")
        
        # Add to the combined list
        combined_data.append(amplitude_data/np.max(amplitude_data))
    
    # Sort by theta values
    sorted_indices = np.argsort(theta_values)
    sorted_data = [combined_data[i] for i in sorted_indices]
    
    # Stack into a 3D array: (number of theta values, data_length_per_receiver, 64 receivers)
    stacked_data = np.stack(sorted_data, axis=0)
    
    # Transpose the array to get (number of theta values, 64 receivers, data_length_per_receiver)
    final_data_array = np.transpose(stacked_data, (0, 2, 1))
    
    return time_receiver, final_data_array


def check_time_consistency(time_receiver):
    # Calculate the difference between consecutive time values
    time_steps = np.diff(time_receiver)
    
    # Check if all time steps are the same
    if np.allclose(time_steps, time_steps[0]):
        print(f"Time steps are consistent: {time_steps[0]}")
    else:
        print("Time steps are inconsistent.")
        
    return time_steps[0]/1e-6 ## micro seconds


# Process all files and get the 3D array
amplitude_receivers = process_files(file_paths)[1]
amplitude_receivers = np.clip(amplitude_receivers[:, :, :], -0.0560, 0.0560)
time_receiver = process_files(file_paths)[0]
dT = check_time_consistency(time_receiver) 

print(f'The shape of processed amplitude data of receiver: {amplitude_receivers.shape}')

plt.plot()
plt.imshow(amplitude_receivers[0, :, :9413].T, aspect="auto")  # Adjusted data
plt.title("B-scan plot ")
plt.show()

############   PWI_TFM Algo ######################

# Processing angles
AS = 60  # Start angle for processing
AE = 61  # End angle for processing
AJ = 1  # Angle step for processing
AN = list((np.arange(AS, AE, AJ)))  # Number of angles for processing
print(f"Number of angles for processing: {len(AN)}")


# Focus points for the +ve angles

# ROI
min_x, max_x = 120,135 #X ROI
min_y, max_y = 0, 28.75
dx = 0.1  # pixel

# Create meshgrid
x1 = np.arange(min_x, max_x, dx)
y1 = np.arange(min_y, max_y, dx)
x, y = np.meshgrid(x1, y1)


# Helper Functions
def generate_receiver_positions(p, N):
    """ Generate receiver positions for processing. """
    return np.arange(0, N, 1) * p


def compute_sample_grid_LDL(t1, r1, x, y, dT,v1=5.760,v2=5.760,v3=5.760):
    """ Precompute the time to go to a point (x, y) and back for each receiver angle. """
    # Extract dimensions
    y_num, x_num = x.shape
    t_len = len(t1)
    r_len = len(r1)

    # Prepare an empty array for SAMPLE
    sample = np.zeros((t_len * r_len, y_num, x_num), dtype=np.int32)

    for idx in prange(t_len * r_len):
        i = idx // r_len  # Index for angle
        j = idx % r_len  # Index for receiver
        t = t1[i]
        r = r1[j]

        # Compute the distances
        x_val = (x-31.5)
        y_val = y
        dist = (np.sqrt((x_val - r) ** 2 + y_val**2))/(v1 * dT) + (y_val * np.cos(t) + ((x_val) * np.sin(t)))/(v2 * dT)

        # Clip the values of dist to have a len time-receiver
        dist_clipped = np.clip(dist, None, len(time_receiver)-1)
        # Convert to int32 type and assign it to SAMPLE
        sample[idx, :, :] = dist_clipped.astype(np.int32)
        
    return np.array(sample)

def compute_sample_grid_LDLL(t1, r1, x, y, dT,v1=5.760,v2=5.760,v3=5.760):
    """ Precompute the time to go to a point (x, y) and back for each receiver angle. """
    # Extract dimensions
    y_num, x_num = x.shape
    t_len = len(t1)
    r_len = len(r1)

    # Prepare an empty array for SAMPLE
    sample = np.zeros((t_len * r_len, y_num, x_num), dtype=np.int32)

    for idx in prange(t_len * r_len):
        i = idx // r_len  # Index for angle
        j = idx % r_len  # Index for receiver
        t = t1[i]
        r = r1[j]

        # Compute the distances
        x_val = (x- (PW / 2))
        y_val = y
        
        gamma = np.arcsin((v2 / v1) * np.sin(t))
        xr = x_val-(thickness-y_val)*np.tan(gamma)
                
        dist = ((x_val)*np.sin(t)+(y_val)*np.cos(t))/(v1*dT) + (np.sqrt((xr - x_val)**2 + (thickness - y_val)**2))/(v2*dT) + (np.sqrt((xr - r) ** 2 + thickness**2))/(v3*dT)

        # Clip the values of dist to have a len time-receiver
        dist_clipped = np.clip(dist, None, len(time_receiver)-1)
        # Convert to int32 type and assign it to SAMPLE
        sample[idx, :, :] = dist_clipped.astype(np.int32)
        
    return np.array(sample)

def compute_sample_grid_LLDL(t1, r1, x, y, dT,v1=5.760,v2=5.760,v3=5.760):
    """ Precompute the time to go to a point (x, y) and back for each receiver angle. """
    # Extract dimensions
    y_num, x_num = x.shape
    t_len = len(t1)
    r_len = len(r1)

    # Prepare an empty array for SAMPLE
    sample = np.zeros((t_len * r_len, y_num, x_num), dtype=np.int32)

    for idx in prange(t_len * r_len):
        i = idx // r_len  # Index for angle
        j = idx % r_len  # Index for receiver
        t = t1[i]
        r = r1[j]

        # Compute the distances
        x_val = (x- (PW / 2))
        y_val = y
        
        gamma = np.arcsin((v2 / v1) * np.sin(t))
        xr = x_val-(thickness-y_val)*np.tan(gamma)
        
        dist = ((xr)*np.sin(t)+(thickness)*np.cos(t))/(v1*dT) + (np.sqrt((xr - x_val)**2 + (thickness - y_val)**2))/(v2*dT) + (np.sqrt((x_val - r) ** 2 + y_val**2))/(v3*dT)
        
        # Clip the values of dist to have a len time-receiver
        dist_clipped = np.clip(dist, None, len(time_receiver)-1)
        # Convert to int32 type and assign it to SAMPLE
        sample[idx, :, :] = dist_clipped.astype(np.int32)
        
    return np.array(sample)

def compute_sample_grid_LLDT(t1, r1, x, y, dT,v1=5.760,v2=5.760,v3=3.118):
    """ Precompute the time to go to a point (x, y) and back for each receiver angle. """
    # Extract dimensions
    y_num, x_num = x.shape
    t_len = len(t1)
    r_len = len(r1)

    # Prepare an empty array for SAMPLE
    sample = np.zeros((t_len * r_len, y_num, x_num), dtype=np.int32)

    for idx in prange(t_len * r_len):
        i = idx // r_len  # Index for angle
        j = idx % r_len  # Index for receiver
        t = t1[i]
        r = r1[j]

        # Compute the distances
        x_val = (x- (PW / 2))
        y_val = y
        
        gamma = np.arcsin((v2 / v1) * np.sin(t))
        xr = x_val-(thickness-y_val)*np.tan(gamma)
        
        dist = ((xr)*np.sin(t)+(thickness)*np.cos(t))/(v1*dT) + (np.sqrt((xr - x_val)**2 + (thickness - y_val)**2))/(v2*dT) + (np.sqrt((x_val - r) ** 2 + y_val**2))/(v3*dT)
        
        # Clip the values of dist to have a len time-receiver
        dist_clipped = np.clip(dist, None, len(time_receiver)-1)
        # Convert to int32 type and assign it to SAMPLE
        sample[idx, :, :] = dist_clipped.astype(np.int32)
        
    return np.array(sample)

def compute_sample_grid_LDTT(t1, r1, x, y, dT,v1=5.760,v2=3.118,v3=3.118):
    """ Precompute the time to go to a point (x, y) and back for each receiver angle. """
    # Extract dimensions
    y_num, x_num = x.shape
    t_len = len(t1)
    r_len = len(r1)

    # Prepare an empty array for SAMPLE
    sample = np.zeros((t_len * r_len, y_num, x_num), dtype=np.int32)

    for idx in prange(t_len * r_len):
        i = idx // r_len  # Index for angle
        j = idx % r_len  # Index for receiver
        t = t1[i]
        r = r1[j]

        # Compute the distances
        x_val = (x- (PW / 2))
        y_val = y
        
        gamma = np.arcsin((v3 / v2) * np.sin(t))
        xr = x_val-(thickness-y_val)*np.tan(gamma)
                
        dist = ((x_val)*np.sin(t)+(y_val)*np.cos(t))/(v1*dT) + (np.sqrt((xr - x_val)**2 + (thickness - y_val)**2))/(v2*dT) + (np.sqrt((xr - r) ** 2 + thickness**2))/(v3*dT)
    
        # Clip the values of dist to have a len time-receiver
        dist_clipped = np.clip(dist, None, len(time_receiver)-1)
        # Convert to int32 type and assign it to SAMPLE
        sample[idx, :, :] = dist_clipped.astype(np.int32)
        
    return np.array(sample)




def compute_image(data, sample, T, AngStart, AngStep, N, positive=True):
    """ 
    Compute the image based on data and precomputed sample distances.
    For positive angles, the data index is (N - j - 1), for negative, it's j.
    """
    image = np.zeros(x.shape)
    for idx, i in enumerate(T):
        b = np.zeros(x.shape)
        for j in range(N):
            if positive:
                a = data[(i - AngStart) // AngStep, j, :]  # For positive angles
            else:
                # For negative angles
                a = data[(i - AngStart) // AngStep, N-j-1, :]
            d = sample[idx * N + j]
            image += a[d]
    return image


def apply_hilbert_transform(image):
    """ Apply Hilbert transform to compute the envelope and magnitude. """
    X_NUM = np.arange(0, image.shape[1], 1)
    image_hilbert = [
        (
            (
                (np.imag(sig.hilbert(image.transpose()[i]))) ** 2
                + (np.real(sig.hilbert(image.transpose()[i]))) ** 2
            )
            ** 0.5
        )
        for i in X_NUM
    ]
    return np.array(image_hilbert).transpose()


def convert_to_db(image):
    """ Convert magnitude to dB scale. """
    image_db = 20 * np.log(image / np.max(image))
    return image_db


def plot_image(image, min_x, max_x, min_y, max_y, cmap, title, vmin=None, vmax=None):
    """ Plot the B-scan image. """
    plt.figure()

    # Check if vmin and vmax are None, otherwise compute them from the image
    if vmin is None:
        vmin = np.min(image)  # Get minimum value of the image
    if vmax is None:
        vmax = np.max(image)  # Get maximum value of the image

    # Plot the image with specified vmin and vmax
    plt.imshow(image, aspect='auto', cmap=cmap, extent=(
        min_x, max_x, max_y, min_y), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.show()

# Main processing function for angles


def process_angles(angles, positive=True):
    """ Process angles either positive or negative """
    r1 = generate_receiver_positions(p, N)

    # Filter angles and convert to radians
    T = [t for t in angles if (t >= 0 if positive else t < 0)]
    # Convert to radians, abs for negative
    t1 = [abs(t * (np.pi / 180)) for t in T]

    print(f"Processing {'positive' if positive else 'negative'} angles:")
    print(
        f"Shape of angle array: {len(t1)}, Shape of receiver positions: {len(r1)}")

    # Precompute sample grid
    sample = compute_sample_grid_LDL(t1, r1, x, y, dT)
    print(f"Shape of sample (distance for each x, y): {np.shape(sample)}")

    # Compute image
    image = compute_image(amplitude_receivers, sample, T, AS, AJ, N, positive)

    # Apply Hilbert transform to get envelope and magnitude
    hilbert_image = apply_hilbert_transform(image)

    # Convert the image to dB scale
    image_db = convert_to_db(hilbert_image)

    # Plot the image
    if not positive:
        image = np.flip(image, axis=1)  # Flip for negative angles
        image_db = np.flip(image_db, axis=1)  # Flip for negative angles

    plot_image(image, min_x, max_x, min_y, max_y, cmap,
               f"B-Scan Image ({'+ve' if positive else '-ve'} angles)")
    plot_image(image_db, min_x, max_x, min_y, max_y, cmap,
               f"B-Scan Image ({'+ve' if positive else '-ve'} angles)", vmin=-30, vmax=0)

    return image


# List of angles to process
AN = list(np.arange(AS, AE, AJ))  # Angle range

# Process for positive angles
image_p = process_angles(AN, positive=True)

# Process for negative angles
image_n = process_angles(AN, positive=False)

combined_img = image_p + image_n

hilbert_image = apply_hilbert_transform(combined_img)
image_db = convert_to_db(hilbert_image)


plot_image(combined_img, min_x, max_x, min_y, max_y, cmap, 'Combined')
plot_image(image_db, min_x, max_x, min_y, max_y,cmap, 'Combined', vmin=-30, vmax=0)
