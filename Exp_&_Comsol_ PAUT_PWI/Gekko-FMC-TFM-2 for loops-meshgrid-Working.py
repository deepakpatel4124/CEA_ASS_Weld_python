#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:15:25 2021

@author: anishkumar
"""
# M2M B-scan data SAFT processing
# Import functions
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig
from joblib import Parallel, delayed
import itertools


# Define custom colormap
color_list = [(0.0, 'white'), (0.33, 'blue'), (0.67, 'yellow'), (1.0, 'red')]
cmap = colors.LinearSegmentedColormap.from_list('mycmap', color_list)

# Enter parameter values
p = 0.5  # pitch in mm
dT = 0.02  # dT in us
dx = 0.1  # dx in mm for both x and y
vel = 5.7  # velocity in mm/us
thick = 50  # thickness in mm

min_x = -40
max_x = 40
min_y = 0
max_y = 55

x_num = int((max_x - min_x) / dx)
y_num = int((max_y - min_y) / dx)

filename = r'C:\Users\Deepak\Desktop\Python FMC-TFM\AM1-FMC-5MHZ-.5P-50MHZ-01_1.capture_acq\data.bin'

# Use memory-mapped file to handle large binary files
data = np.memmap(filename, dtype='<i2', mode='r')

dL = data[63] + 10  # data_points
N = int(data[55])  # transducer_num

# Extract and reshape the relevant portion of data
data = data[64:].clip(-512, 512)[:N * (N * dL + 10)].reshape(N, (N * dL + 10))[:, :N * dL].reshape(N, N, dL)[:, :, :dL - 10]

min_T = -(N / 2 - 0.5) * p

# Generate meshgrid
y1 = np.arange(min_y, max_y, dx)
x1 = np.arange(min_x, max_x, dx)
x, y = np.meshgrid(x1, y1)

T = np.arange(0, N)
t1 = (T * p + min_T)

R = np.arange(0, N)
r1 = (R * p + min_T)

# Prepare the image array
image = np.zeros((y_num, x_num), dtype=np.float32)

print('1')

# Start timing
tic = time.time()


# Function to compute SAMPLE for a given pair (t, r)
def compute_sample(t, r):
    return (((np.sqrt((np.square(x - r)) + np.square(y))) + (np.sqrt((np.square(x - t)) + np.square(y)))) / (vel * dT)).astype(int)

# Parallel processing of SAMPLE calculation
results = Parallel(n_jobs=-1)(delayed(compute_sample)(t, r) for t, r in itertools.product(t1, r1))


# Convert results to numpy array
SAMPLE = np.array(results).reshape(N, N, y_num, x_num)

print('2')

# Accumulate image using vectorized operations
for g in range(N):
    for h in range(N):
        indices = SAMPLE[g, h, :, :]
        np.add.at(image, (slice(None), slice(None)), data[g, h, indices])

image = image.transpose()

print('3')

# Hilbert transform and plotting
image1 = np.abs(sig.hilbert(image, axis=1))
image1 = image1 / np.max(image1)
image2 = 20 * np.log10(image1 + 1e-12)  # Add small value to avoid log(0)

plt.subplot(2, 1, 1)
plt.imshow(image1.transpose(), aspect='auto', cmap=cmap, vmin=0, vmax=0.5)
plt.xticks([0, (max_x - min_x) / (2 * dx), (max_x - min_x) / dx], [min_x, (min_x + max_x) / 2, max_x])
plt.yticks([0, (max_y - min_y) / (2 * dx), (max_y - min_y) / dx], [min_y, (min_y + max_y) / 2, max_y])
plt.colorbar()
plt.title(filename)

plt.subplot(2, 1, 2)
plt.imshow(image2.transpose(), aspect='auto', cmap=cmap, vmin=-80, vmax=0)
plt.colorbar()

# End timing and print duration
toc = time.time()
print(toc - tic)

plt.show()
