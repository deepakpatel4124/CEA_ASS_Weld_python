import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig
import itertools
from tqdm.auto import tqdm
from numba import njit, prange
import scipy.signal as signal 

filename =  r"PWI_data\PWI_MultiModal_data\01_10_2024\PWI_PM_MM_WC_ROOT_Fire_C_-89 to 0.capture_acq/data.bin"

AngStart = -89
AngEnd = 0
AngStep = 1


p = 1.0  # pitch in mm
dT = 0.02  # dT in us
v_l = 5.760  # Logitudional velocity in mm/us
v_s = 3.118  # Shear velocity in mm/us
mmdp = v_l * dT / 2  # calculated parameter


color_list = [
    (0.0, "white"),  # white
    (0.33, "blue"),  # blue
    (0.67, "yellow"),  # yellow
    (1.0, "red"),  # red
]
cmap = colors.LinearSegmentedColormap.from_list("mycmap", color_list)

########## Receiver Data Correction ##########

data = np.fromfile(filename, dtype="<i2", sep="")
dL = int(data[63] + 10)  # data_points
N = int(data[55])  # transducer_num
print(f"Data Length: {dL}, Transducer Count: {N}")


M = 1 + int((AngEnd - AngStart) / AngStep)
PW = (N - 1) * p  # PW is the probe width
data = (
    data[64:]
    .clip(-512, 512)[: M * (N * dL + 10)]
    .reshape(M, (N * dL + 10))[:, : N * dL]
).reshape(M, N, dL)[
    :, :, : dL - 10
]  # Clipping the data to 10 bit i.e. +-512


# data length to be deleted from every A-scan of kth angle with respect to the max angle data
data1 = np.zeros((M, N, dL - 10))
for i in range(M):
    theta = (i * AngStep + AngStart) * np.pi / 180
    D1 = int(
        PW * (((np.sin(max(abs(AngEnd), abs(AngStart)) * np.pi / 180) -
              np.sin(np.abs(theta))) / 2) / (2 * mmdp))
    )
    data1[i, :, : dL - 10 - D1] = data[i, :, D1:]

data1 = data1/np.max(data1)

# Plot raw data and adjusted data for a specific angle (63rd angle)
plt.subplot(1, 2, 1)
plt.imshow(np.transpose(data[50, :, :]), aspect="auto")  # Original data
plt.title("Original Data ")

plt.subplot(1, 2, 2)
plt.imshow(np.transpose(data1[80, :, :]), aspect="auto")  # Adjusted data
plt.title("Receiver correction ")
plt.show()

plt.plot(data1[30,31,:])



####### Thickness Calculation  ##############

def calculate_thickness(data1, data, zero_deg = int(abs(AngStart)), mid_transducer = int((N/2)-1), len_0=400, dT=0.02,threshold = 1, v_l = 5.760):
    # Extract signal and normalize it
    sig = data1[zero_deg, mid_transducer, :] / np.max(data1[zero_deg, mid_transducer, :])
    
    # Plot the full signal
    plt.plot(sig, linestyle='-', linewidth=0.5, color='b')
    
    # Add vertical lines to show the start and end of the two segments
    plt.axvline(x=len_0, color='r', linestyle='--', label=f'Segment 1 Start ({len_0})')
    plt.axvline(x=2*len_0, color='g', linestyle='--', label=f'Segment 1 End / Segment 2 Start ({2*len_0})')
    plt.axvline(x=3*len_0, color='r', linestyle='--', label=f'Segment 2 End ({3*len_0})')
    
    
    plt.xlabel('Time, data point')
    plt.ylabel('Amplitude, A.U.')
    plt.show()

    # Create subplots for the two signal segments
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))  # Two rows, one column

    # Plot first segment
    axs[0].plot(sig[len_0:2*len_0], linestyle='-', linewidth=0.5, color='b')
    axs[0].set_xlabel('Time, data point')
    axs[0].set_ylabel('Amplitude, A.U.')
    axs[0].set_title('First Signal Segment')

    # Plot second segment
    axs[1].plot(sig[2*len_0:3*len_0], linestyle='-', linewidth=0.5, color='b')
    axs[1].set_xlabel('Time, data point')
    axs[1].set_ylabel('Amplitude, A.U.')
    axs[1].set_title('Second Signal Segment')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # Perform cross-correlation of the two segments
    corr = signal.correlate(sig[len_0:2*len_0], sig[2*len_0:3*len_0])
    
    # Find the position of the maximum correlation
    max_p = np.argmax(corr)
    
    # Calculate time of flight (tof_0)
    tof_0 = 2 * len_0 - max_p
    
    # Calculate thickness
    thickness = (tof_0 * dT) * v_l / 2
    print(f"Using cross-Corelation the thickness is: {thickness}")
    
    
    # Find the first crossing of the threshold in each segment
    def find_first_crossing(signal_segment, threshold):
        for idx, value in enumerate(signal_segment):
            if value >= threshold:
                return idx
        return None

    # Find the first threshold crossing for each segment
    first_cross_1 = find_first_crossing(sig[len_0:2*len_0], threshold)
    first_cross_2 = find_first_crossing(sig[2*len_0:3*len_0], threshold)
    tof_0_threshold = (len_0-first_cross_1)+first_cross_2
    print(f"Using threshold the thickness is: {(tof_0_threshold* dT) * v_l / 2}")


    return thickness, tof_0

thickness = calculate_thickness(data1, data, len_0=400)[0]

############   PWI_TFM Algo ######################

# Processing angles
AS = -85  # Start angle for processing
AE = -0  # End angle for processing
AJ = 5 # Angle step for processing
AN = list((np.arange(AS, AE, AJ)))  # Number of angles for processing
print(f"Number of angles for processing: {len(AN)}")


# ROI
min_x, max_x = 80,110 #X ROI
min_y, max_y = 0, thickness   #Y ROI
dx = 0.25 # pixel

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
        dist_clipped = np.clip(dist, None, dL - 20)
        # Convert to int32 type and assign it to sample
        sample[idx, :, :] = dist_clipped.astype(np.int32)

    return np.array(sample)


def compute_image(data, sample, T, AngStart, AngStep, N, angle_sign):
    """ 
    Compute the image based on data and precomputed sample distances.
    For positive angles, the data index is (N - j - 1), for negative, it's j.
    """
    image = np.zeros(x.shape)
    for idx, i in enumerate(T):
        b = np.zeros(x.shape)
        for j in range(N):
            if angle_sign == 'positive':
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
        (((np.imag(sig.hilbert(image.transpose()[i]))) ** 2+ (np.real(sig.hilbert(image.transpose()[i]))) ** 2)** 0.5)
        for i in X_NUM
    ]
    return np.array(image_hilbert).transpose()



def convert_to_db(image, norm =30):
    if norm ==None:
        norm = np.max(image)
    """ Convert magnitude to dB scale. """
    image_db = 20 * np.log(image /norm)
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
         max_x,min_x, min_y, max_y), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.show()


def process_angles(AS,AE,AJ,N,p,mode = 'LDL'):
    
    angles = list(np.arange(AS, AE, AJ))  # Angle range

    r1 = generate_receiver_positions(N,p)

    """Process angles based on angle_sign: 'positive', 'negative', or 'both'."""
    
    # Define angle range based on the provided parameters (AS, AE, AJ assumed to be global or passed in)
    angles = list(np.arange(AS, AE+1, AJ))  # Angle range

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
    image_pos = compute_image(data1, sample_pos, T_pos, AngStart, AngStep, N, 'positive')
    
    image_neg = compute_image(data1, sample_neg, T_neg, AngStart, AngStep, N, 'negative')
    # Flip negative image for correct orientation
    # image_neg = np.flip(image_neg, axis=1)
    
    combined_image = image_pos+image_neg
    combined_hilbert_image = apply_hilbert_transform(combined_image)
    combined_image_db = convert_to_db(combined_hilbert_image)
    
    plot_image(combined_image_db, min_x, max_x, min_y, max_y, cmap, "B-Scan Image (both angles)", vmin=-30, vmax=0)

    return combined_image_db

image_db_LLDL = process_angles(AS,AE,AJ,N,p,mode = 'LDL')






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
    image_pos = compute_image(data1, sample_pos, T_pos, AngStart, AngStep, N, 'positive')
    image_neg = compute_image(data1, sample_neg, T_neg, AngStart, AngStep, N, 'negative')

    # Flip the negative image horizontally for correct orientation
    # image_neg = np.flip(image_neg, axis=1)

    # Combine the positive and negative images
    combined_image = image_pos + image_neg

    # Apply Hilbert transform and convert the combined image to dB scale
    combined_hilbert_image = apply_hilbert_transform(combined_image)
    combined_image_db = convert_to_db(combined_hilbert_image)

    # Store the combined image in the list
    images_list.append(combined_image_db)

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
plt.imshow(final_db, cmap=cmap, aspect='auto', extent=[min_x, max_x, min_y, max_y], vmin=-30, vmax=10)
plt.title("Final Weighted Sum of B-Scan Images")
plt.colorbar(label='Amplitude (dB)')
plt.show()















vmin = -30
vmax =0

image_data = image_db_LLDL

# Define the region in pixel (start_row, end_row, start_col, end_col)
region1 = np.array([30, 85, 50, 90])
region2 = np.array([0, 30, 50, 90])


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
region1 = np.array([0, 75, 50, 90])
# region1 = np.array([30, 114, 40, 75])
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
    min_crossing= max(all_crossings, key=lambda x: x[0])

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










