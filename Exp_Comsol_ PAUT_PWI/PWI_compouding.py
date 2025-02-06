import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig
import itertools
from numba import njit, prange

filename = r"PWI_data\PWI PM WP#0_WC_SDHS\WP#0 PM WC root -60 to 60 deg step 1 2.25 50Mhz 30dB.capture_acq\data.bin"

AngStart = -60
AngEnd = 60
AngStep = 1


p = 1.0  # pitch in mm
dT = 0.02  # dT in us
vel = 5.760  # velocity in mm/us
mmdp = vel * dT / 2  # calculated parameter


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

# Plot raw data and adjusted data for a specific angle (63rd angle)
plt.subplot(1, 2, 1)
plt.imshow(np.transpose(data[44, :, :500]), aspect="auto")  # Original data
plt.title("Original Data ")

plt.subplot(1, 2, 2)
plt.imshow(np.transpose(data1[44, :, :500]), aspect="auto")  # Adjusted data
plt.title("Receiver correction ")
plt.show()


############   PWI_TFM Algo ######################

# Processing angles
AS = -60  # Start angle for processing
AE = 60  # End angle for processing
AJ = 1  # Angle step for processing
AN = list((np.arange(AS, AE, AJ)))  # Number of angles for processing
print(f"Number of angles for processing: {len(AN)}")


# Focus points for the +ve angles

# ROI
min_x, max_x = 0, 64
min_y, max_y = 0, 35
dx = 0.25  # pixel

# Create meshgrid
x1 = np.arange(min_x, max_x, dx)
y1 = np.arange(min_y, max_y, dx)
x, y = np.meshgrid(x1, y1)

# Helper Functions


def generate_receiver_positions(p, N):
    """ Generate receiver positions for processing. """
    return np.arange(N - 1, -1, -1) * p


def compute_sample_grid(t1, r1, x, y, vel, dT):
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
        dist = (
            np.sqrt((x_val - r) ** 2 + y_val**2)
            + (y_val * np.cos(t) + ((x_val) * np.sin(t)))
        ) / (vel * dT)

        # Clip the values of dist to have a len time-receiver
        dist_clipped = np.clip(dist, None, dL-20)
        # Convert to int32 type and assign it to sample
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
                a = data[(i - AngStart) // AngStep, N -
                         j - 1, :]  # For positive angles
            else:
                # For negative angles
                a = data[(i - AngStart) // AngStep, j, :]
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
    sample = compute_sample_grid(t1, r1, x, y, vel, dT)
    print(f"Shape of sample (distance for each x, y): {np.shape(sample)}")

    # Compute image
    image = compute_image(data1, sample, T, AS, AJ, N, positive)

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
plot_image(image_db, min_x, max_x, min_y, max_y,
           cmap, 'Combined', vmin=-30, vmax=0)
