import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig
import itertools
from tqdm.auto import tqdm
from numba import njit, prange


# List of file paths (add all your file paths here)
# file_paths = [r'PWI_data\PWI_Simulation\PWI_PM_Defect_MM_WC\PWI_PM_Defect_MM_WC_theta_60.txt']

# Function to load text files within a Folder
def load_txt_files(folder_path):
    file_paths = []
    # Iterate through all the files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_paths.append(file_path)
            sorted_files = sorted(file_paths, key=extract_theta_from_filename)
    return sorted_files

# Function to extract the theta value from the filename
def extract_theta_from_filename(filename):
    # Assuming the filename is in the format: "PM_Defect_{number}_theta_{theta_value}.txt"
    basename = os.path.basename(filename)  # Get only the file name, not the path
    theta_part = basename.split('_')[-1]   # Extract the last part which contains theta value and .txt
    theta_value = theta_part.replace('.txt', '')  # Remove '.txt' from theta part
    return float(theta_value)


folder_path = r'F:\PWI_PM\PWI_PM_Defect_MM_WC_Root_Fire_C'
file_paths = load_txt_files(folder_path)


########### Define the parameters ###########

AngStart = 0
AngEnd = 85
AngStep = 5

# Parameter definitions
p = 1.0  # pitch in mm
N = 64
v_l = 5.760  # Logitudional velocity in mm/us
v_s = 3.118  # Shear velocity in mm/us
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

    # Clean the data with a progress bar
    cleaned_data = []
    for line in tqdm(data_lines, desc="Processing data", unit=" lines"):
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

# Beow two functions are to have the same sampling rate as of exp
point_to_average = 14

def average_every_n_points(data, n):
    data_trimmed = data[:len(data) - len(data) % n]  # Trim excess points to make the length divisible by 'n'
    averaged_data = np.mean(data_trimmed.reshape(-1, n), axis=1)  # Average every 'n' points
    return averaged_data

def average_every_n_points_2d(data, n):
    # Trim excess points to make the length divisible by 'n'
    data_trimmed = data[:len(data) - len(data) % n]
    # Reshape and average over every 'n' points along the row axis
    averaged_data = np.mean(data_trimmed.reshape(-1, n, data.shape[1]), axis=1)
    return averaged_data


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
        current_time_receiver = average_every_n_points(current_time_receiver, point_to_average)
        
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
        amplitude_data = average_every_n_points_2d(amplitude_data, point_to_average)
        
        
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
time_receiver, amplitude_receivers = process_files(file_paths)
dT = check_time_consistency(time_receiver) 


amplitude_receivers_scaled = np.clip(amplitude_receivers[:, :, :], -0.1560, 0.15660)/0.15660

# alpha = 0.1 # attenuation coefficient

# amplitude_receivers_scaled_attenuated  = np.zeros_like(amplitude_receivers_scaled)
# for i in range(amplitude_receivers_scaled_attenuated.shape[1]):
#     indices = np.arange(amplitude_receivers_scaled_attenuated.shape[2]) 
#     amplitude_receivers_scaled_attenuated[:,i,:] = amplitude_receivers_scaled[:,i,:]*np.exp(-alpha * indices*dT)


print(f'The shape of processed amplitude data of receiver: {amplitude_receivers.shape}')

plt.plot()
plt.imshow(amplitude_receivers_scaled[0, :, :].T, aspect="auto")  # Adjusted data
plt.title("B-scan plot ")
plt.show()

plt.plot(amplitude_receivers_scaled[0,31,:2500])


############   PWI_TFM Algo ######################

# Processing angles
AS = 50  # Start angle for processing
AE = 80  # End angle for processing
AJ = 5  # Angle step for processing
AN = list((np.arange(AS, AE, AJ)))  # Number of angles for processing
print(f"Number of angles for processing: {len(AN)}")


# Focus points for the +ve angles

# ROI
min_x, max_x = 80,110 #X ROI
min_y, max_y = 0, thickness
dx = 0.25  # pixel

# Create meshgrid
x1 = np.arange(min_x, max_x, dx)
y1 = np.arange(min_y, max_y, dx)
x, y = np.meshgrid(x1, y1)


# Helper Functions
def generate_receiver_positions(N, p):
    """ Generate receiver positions for processing. """
    return np.arange(0, N, 1) * p

# Helper function to assign velocities based on mode
def assign_velocities(mode):
    velocities = []
    for char in mode:
        if char == 'L':
            velocities.append(v_l)
        elif char == 'T':
            velocities.append(v_s)
        elif char == 'D':
            continue
        else:
            raise ValueError("Mode contains invalid character (use only 'L', 'D'  or 'T')")
    return velocities

def compute_sample_grid_PWI(t1, r1, x, y, dT, mode='LDL'):
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
        x_val = x
        y_val = y

        # Assign velocities based on the mode
        if mode in ['LDL', 'LDT', 'TDL', 'TDT']:
            v1, v2 = assign_velocities(mode)  # Here, v1 = v_l, v2 = v_s
            dist = (y_val * np.cos(t) + (x_val * np.sin(t))) / (v1 * dT) + (np.sqrt((x_val - r) ** 2 + y_val ** 2)) / (v2 * dT)
            
        elif mode in ['LLDLL']:
            v1, v2, v3, v4 = assign_velocities(mode)  # Assign v1, v2, v3 dynamically based on 'LLDL' or 'LLDT'
       
            dist = ((2*thickness-y_val) * np.cos(t) + (x_val * np.sin(t))) / (v1 * dT) + (np.sqrt((x_val - r) ** 2 + (2*thickness-y_val) ** 2)) / (v2 * dT)   
        
        elif mode in ['LLDL','LLDT','LTDL','LTDT','TLDL','TLDT','TTDL','TTDT']:
            v1, v2, v3 = assign_velocities(mode)  # Assign v1, v2, v3 dynamically based on 'LLDL' or 'LLDT'
            gamma = np.arcsin((v2 / v1) * np.sin(t))
            xr = x_val - (thickness - y_val) * np.tan(gamma)

            dist = ((xr) * np.sin(t) + (thickness) * np.cos(t)) / (v1 * dT) + (np.sqrt((xr - x_val) ** 2
                   + (thickness - y_val) ** 2)) / (v2 * dT) + (np.sqrt((x_val - r) ** 2 + y_val ** 2)) / (v3 * dT)

        elif mode in ['LDLL','LDLT','LDTL','LDTT','TDLL','TDLT','TDTL','TDTT']:
            v1, v2, v3 = assign_velocities(mode)  # Assign v1, v2, v3 dynamically based on 'LDLL' or 'LDTT'
                
            dist = ((x_val)*np.sin(t)+(y_val)*np.cos(t))/(v1*dT) + (np.sqrt((x_val - r) ** 2 + (2*thickness-y_val) ** 2)) / (v2 * dT)
        
        elif mode in ['LLDLLL']:
            v1, v2, v3,v4,v5 = assign_velocities(mode)  # Assign v1, v2, v3 dynamically based on 'LDLL' or 'LDTT'
            
            gamma = np.arcsin((v3 / v2) * np.sin(t))
            xr = x_val-((2*thickness)-(2*thickness-y_val))*np.tan(gamma)
                
            dist = ((x_val)*np.sin(t)+((2*thickness-y_val))*np.cos(t))/(v1*dT) + (np.sqrt((xr - x_val)**2 +((2*thickness) - (2*thickness-y_val))**2))/(v2*dT) + (np.sqrt((xr - r) ** 2 + (2*thickness)**2))/(v3*dT)


        else:
            print("Incorrect Mode Entered")

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
            ((np.imag(sig.hilbert(image.transpose()[i]))) ** 2+ (np.real(sig.hilbert(image.transpose()[i]))) ** 2)** 0.5
        )
        for i in X_NUM
    ]
    return np.array(image_hilbert).transpose()


def convert_to_db(image,norm =30):
    if norm == None:
        norm = np.max(image)
    """ Convert magnitude to dB scale. """
    image_db = 20 * np.log(image / norm)
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






def process_angles(AS,AE,AJ,N,p,mode = 'LDL'):
    
    angles = list(np.arange(AS, AE, AJ))  # Angle range

    r1 = generate_receiver_positions(N,p)

    """Process angles based on angle_sign: 'positive', 'negative', or 'both'."""
    
    # Define angle range based on the provided parameters (AS, AE, AJ assumed to be global or passed in)
    angles = list(np.arange(AS, AE, AJ))  # Angle range

    # Generate receiver positions
    r1 = generate_receiver_positions(N, p)

    # Process both positive and negative angles separately
    T_pos = [t for t in angles if t >= 0]
    T_neg = [t for t in angles if t < 0]
    t1_pos = [t * (np.pi / 180) for t in T_pos]
    t1_neg = [abs(t * (np.pi / 180)) for t in T_neg]

    print("Processing both positive and negative angles:")
    # Precompute sample grids for both positive and negative angles
    sample_pos = compute_sample_grid_PWI(t1_pos, r1, x, y, dT, mode=mode)
    sample_neg = compute_sample_grid_PWI(t1_neg, r1, x, y, dT, mode=mode)

    print(f"Shape of positive sample: {np.shape(sample_pos)}, negative sample: {np.shape(sample_neg)}")

    # Compute images for positive and negative angles
    image_pos = compute_image(amplitude_receivers_scaled, sample_pos, T_pos, AngStart, AngStep, N, 'positive')
    
    image_neg = compute_image(amplitude_receivers_scaled, sample_neg, T_neg, AngStart, AngStep, N, 'negative')
    # Flip negative image for correct orientation
    image_neg = np.flip(image_neg, axis=1)
    
    combined_image = image_pos+image_neg
    combined_hilbert_image = apply_hilbert_transform(combined_image)
    combined_image_db = convert_to_db(combined_hilbert_image)
    
    plot_image(combined_image_db, min_x, max_x, min_y, max_y, cmap, "B-Scan Image (both angles)", vmin=-30, vmax=0)

    return combined_image_db

image_db_LLDL = process_angles(AS,AE,AJ,N,p,mode = 'LDTT')







# Define the list of modes
modes = ['LDL', 'LDLL', 'LDTT', 'LLDLL', 'LLDL', 'LLDT', 'LLDLLL', 'LTDL', 'LTDT']

# List of angles to process
angles = list(np.arange(AS, AE, AJ))  # Angle range

r1 = generate_receiver_positions(N,p)

# Filter angles and convert to radians
T = [t for t in angles]
# Convert to radians, abs for negative
t1 = [abs(t * (np.pi / 180)) for t in T]

# Set up a 3x3 grid for plotting
fig, axs = plt.subplots(3, 3, figsize=(12, 8))

# Initialize an empty list to store the images
images_list = []

# Loop through each mode and compute the images
for idx, mode in enumerate(tqdm(modes, desc="Processing Modes")):
    
    # Filter and convert angles for positive and negative
    T_pos = [t for t in angles if t >= 0]
    T_neg = [t for t in angles if t < 0]
    t1_pos = [t * (np.pi / 180) for t in T_pos]  # Positive angles to radians
    t1_neg = [abs(t * (np.pi / 180)) for t in T_neg]  # Negative angles to radians (absolute)

    # Precompute sample grids for both positive and negative angles using the current mode
    sample_pos = compute_sample_grid_PWI(t1_pos, r1, x, y, dT, mode=mode)
    sample_neg = compute_sample_grid_PWI(t1_neg, r1, x, y, dT, mode=mode)

    # Compute images for positive and negative angles
    image_pos = compute_image(amplitude_receivers_scaled, sample_pos, T_pos, AngStart, AngStep, N, 'positive')
    image_neg = compute_image(amplitude_receivers_scaled, sample_neg, T_neg, AngStart, AngStep, N, 'negative')

    # Flip the negative image horizontally for correct orientation
    # image_neg = np.flip(image_neg, axis=1)

    # Combine the positive and negative images
    combined_image = image_pos + image_neg

    # Store the combined image in the list
    images_list.append(combined_image)

    # Apply Hilbert transform and convert the combined image to dB scale
    combined_hilbert_image = apply_hilbert_transform(combined_image)
    combined_image_db = convert_to_db(combined_hilbert_image)


    # Get row and column index for the current subplot
    row = idx // 3
    col = idx % 3

    # Plot the combined image in dB scale in the appropriate subplot
    axs[row, col].imshow(combined_image_db, cmap=cmap, aspect='auto', 
                         extent=[min_x, max_x, min_y, max_y], vmin=-30, vmax=0)
    axs[row, col].set_title(f"Mode: {mode}")
    axs[row, col].axis('on')  # You can set this to 'off' if you want to hide the axes

# Adjust layout to avoid overlap
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Adjust space at the top for the main title

# Add a global title (optional)
plt.suptitle("Combined B-Scan Images for Different Modes", fontsize=16)

# Show the plot
plt.show()



# Now, sum all the images with their respective weights
weights = [0.9, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.3]  # Example of weighting factors
final_image = np.zeros_like(images_list[0])

# Apply the weights and sum the images
for i, img in enumerate(images_list):
    final_image += weights[i] * img  # Weighted sum of the images

final_hilbert= apply_hilbert_transform(final_image)
final_db = convert_to_db(final_hilbert)

# Plot the final weighted sum image
plt.figure(figsize=(8, 6))
plt.imshow(final_db, cmap=cmap, aspect='auto', extent=[min_x, max_x, min_y, max_y], vmin=-30, vmax=0)
plt.title("Final Weighted Sum of B-Scan Images")
plt.colorbar(label='Amplitude (dB)')
plt.show()










vmin = -30
vmax =0

image_data = image_db_LLDL

# Define the region in pixel (start_row, end_row, start_col, end_col)
region1 = np.array([30, 100, 40, 80])
region2 = np.array([100, 116, 40, 80])


def pixel_to_physical1(pixel_region):
    x_pixel ,y_pixel = pixel_region
    
    # Convert x coordinates
    x_physical = round((min_y + (x_pixel / image_data.shape[0]) * (max_y - min_y)),2)
    
    # Convert y coordinates
    y_physical = round((min_x + (y_pixel / image_data.shape[1]) * (max_x - min_x)),2)

    return (y_physical , x_physical)


# Extract the region from the image data for the first region
region1_data = image_data[region1[0]:region1[1], region1[2]:region1[3]]
region2_data = image_data[region2[0]:region2[1], region2[2]:region2[3]]

# Find the location of the max value in the first region
max_loc_in_region1 = np.unravel_index(np.argmax(region1_data), region1_data.shape)
max_loc_in_region2 = np.unravel_index(np.argmax(region2_data), region2_data.shape)

# Adjust max location to fit the original image coordinates for region1
max_loc1 = (max_loc_in_region1[0] + region1[0], max_loc_in_region1[1] + region1[2])
max_loc2 = (max_loc_in_region2[0] + region2[0], max_loc_in_region2[1] + region2[2])

#Draw vertical and horizontal lines at the max value location for region1
plt.axhline(y=max_loc1[0], color='r', linestyle='--')  # Horizontal line for region1
plt.axvline(x=max_loc1[1], color='r', linestyle='--')  # Vertical line for region1
plt.scatter(max_loc1[1], max_loc1[0], color='blue', s=100, marker='o')  # Marker for region1

#Draw vertical and horizontal lines at the max value location for region2
plt.axhline(y=max_loc2[0], color='black', linestyle='--')  # Horizontal line for region2
plt.axvline(x=max_loc2[1], color='black', linestyle='--')  # Vertical line for region2
plt.scatter(max_loc2[1], max_loc2[0], color='black', s=100, marker='o')  # Marker for region2

#Plot the region boundaries (for visualization)
rect1 = plt.Rectangle((region1[2], region1[0]), region1[3]-region1[2], region1[1]-region1[0], 
                      edgecolor='green', facecolor='none', linewidth=2)
plt.gca().add_patch(rect1)

rect2 = plt.Rectangle((region2[2], region2[0]), region2[3]-region2[2], region2[1]-region2[0], 
                      edgecolor='purple', facecolor='none', linewidth=2)
plt.gca().add_patch(rect2)

plt.text(region1[2], region1[0]-3, 'Region-1', color='green', fontsize=10)  # Label for Region-1
plt.text(region2[2], region2[0]-3, 'Region-2', color='purple', fontsize=10)  # Label for Region-2
plt.title(f"Region1 max at {pixel_to_physical1(max_loc1)}, Value: {round(image_data[(max_loc1)])}\n"
          f"Region2 max at {pixel_to_physical1(max_loc2)}, Value: {round(image_data[max_loc2])}")

plt.imshow(image_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

# Set the new x-axis ticks and labels
original_x_ticks = np.linspace(0, image_data.shape[1], num=5)  # Original pixel tick positions
new_x_labels = np.linspace(min_x, max_x, num=5)  # Corresponding physical labels from 80 to 110

# Set the new y-axis ticks and labels
original_y_ticks = np.linspace(0, image_data.shape[0], num=5)  # Original pixel tick positions
new_y_labels = np.linspace(min_y, max_y, num=5)  # Corresponding physical labels from 0 to thickness

# Set the new ticks and labels
plt.xticks(original_x_ticks, np.round(new_x_labels,3))  # Use the new labels for x ticks
plt.yticks(original_y_ticks, np.round(new_y_labels,3))  # Use the new labels for y ticks

# Labels and title
plt.xlabel('X-axis (ROI)')
plt.ylabel('Y-axis (Thickness)')
plt.colorbar(label='dB')  # Add a colorbar to indicate intensity
plt.grid(True)  # Optional: Show grid


plt.show()

















image_data = image_db_LLDL

# Define the region in pixel (start_row, end_row, start_col, end_col)
region1 = np.array([35, 116, 45, 75])
# region1 = np.array([0, 75, 45, 70])
# region1 = np.array([0, 75, 40, 75])

# Set the threshold for the dB value (for example, -30 dB)
threshold = -30

# Define the region in pixel (start_row, end_row, start_col, end_col)
region1_data = image_data[region1[0]:region1[1], region1[2]:region1[3]]

# Find the location of the max value in the first region
max_loc_in_region1 = np.unravel_index(np.argmax(region1_data), region1_data.shape)

# Adjust max location to fit the original image coordinates for region1
max_loc1 = (max_loc_in_region1[0] + region1[0], max_loc_in_region1[1] + region1[2])

#Draw vertical and horizontal lines at the max value location for region1
plt.axhline(y=max_loc1[0], color='r', linestyle='--')  # Horizontal line for region1
plt.axvline(x=max_loc1[1], color='r', linestyle='--')  # Vertical line for region1
plt.scatter(max_loc1[1], max_loc1[0], color='blue', s=100, marker='o')  # Marker for region1

#Plot the region boundaries (for visualization)
rect1 = plt.Rectangle((region1[2], region1[0]), region1[3]-region1[2], region1[1]-region1[0], 
                      edgecolor='green', facecolor='none', linewidth=2)
plt.gca().add_patch(rect1)

def pixel_to_physical2(pixel_region):
    x_pixel ,y_pixel = pixel_region
    
    # Convert x coordinates
    y_physical = round((min_y + (y_pixel / image_data.shape[0]) * (max_y - min_y)),2)
    
    # Convert y coordinates
    x_physical = round((min_x + (x_pixel / image_data.shape[1]) * (max_x - min_x)),2)

    return (x_physical , y_physical)

# Function to find the row where signal crosses the threshold for each column
def find_all_threshold_crossings(region_data, threshold_value):
    # Initialize a list to store the first row crossing for each column
    all_crossings = []
    
    # Loop through each column in the region data
    for col in range(region_data.shape[1]):
        # Find the row indices where the value crosses the threshold
        row_indices = np.where(region_data[:, col] >= threshold_value)[0]
        
        # If any crossing is found, append the smallest row (first crossing)
        if len(row_indices) > 0:
            all_crossings.append((row_indices[0], col))  # Append (row, col) tuple for the first crossing
    
    # If no crossing is found in any column, return None
    if len(all_crossings) == 0:
        return None
    
    # Find the minimum row crossing among all columns
    min_crossing= min(all_crossings, key=lambda x: x[0])

    return all_crossings, min_crossing  # Return all crossings and the minimum row crossing


# Find the first threshold crossing row for Region 1
first_crossing = find_all_threshold_crossings(region1_data, threshold)[1]

# Plot the first threshold crossing for Region 1 if found
if first_crossing is not None:
    row, col = first_crossing
    # Adjust to fit the original image coordinates
    crossing_loc = (row + region1[0], col + region1[2])
    
    # Plot horizontal line and marker at the crossing point
    plt.axhline(y=crossing_loc[0], color='blue', linestyle=':', alpha=1)  # Horizontal line
    plt.scatter(crossing_loc[1], crossing_loc[0], color='red', s=60, marker='.')  # Marker at crossing point
    
    # Display the physical coordinates of the threshold crossing
    physical_coords = pixel_to_physical2((crossing_loc[1], crossing_loc[0]))
    plt.text(crossing_loc[1], crossing_loc[0]-2, f'{physical_coords}', color='red', fontsize=10)

# Continue with your existing plotting code
plt.title(f"Region -> max at {pixel_to_physical1(max_loc1)}, Value: {round(image_data[(max_loc1)])}")

plt.imshow(image_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

# Set the new x-axis ticks and labels
original_x_ticks = np.linspace(0, image_data.shape[1], num=5)  # Original pixel tick positions
new_x_labels = np.linspace(min_x, max_x, num=5)  # Corresponding physical labels from 80 to 110

# Set the new y-axis ticks and labels
original_y_ticks = np.linspace(0, image_data.shape[0], num=5)  # Original pixel tick positions
new_y_labels = np.linspace(min_y, max_y, num=5)  # Corresponding physical labels from 0 to thickness

# Set the new ticks and labels
plt.xticks(original_x_ticks, np.round(new_x_labels,3))  # Use the new labels for x ticks
plt.yticks(original_y_ticks, np.round(new_y_labels,3))  # Use the new labels for y ticks

# Labels and title
plt.xlabel('X-axis (ROI)')
plt.ylabel('Y-axis (Thickness)')
plt.colorbar(label='dB')  # Add a colorbar to indicate intensity
plt.grid(True)  # Optional: Show grid


plt.show()













