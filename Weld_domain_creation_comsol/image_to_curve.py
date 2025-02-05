from skimage import io, measure, morphology
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

scale_factor = 0.1  # mm per pixel (5mm/500pixel)

# Load the image
image = io.imread('weld_profile.png', as_gray=True)

# Create a binary mask of features
mask = image == image.min()

# Close holes in features
mask = morphology.binary_closing(mask, morphology.square(3))

skeleton = morphology.skeletonize(mask)

# Display the mask
plt.matshow(skeleton, cmap='gray')
plt.show()

# Get the coordinates of each feature
rp = measure.regionprops(measure.label(skeleton))

# Get coordinates of one of the features
coords = rp[0].coords

# Sort coordinates by x-value
sorted_indices = np.argsort(coords[:, 1])
x_sorted = coords[sorted_indices, 0] * scale_factor
y_sorted = coords[sorted_indices, 1] * scale_factor

# plt.plot(y_sorted, -x_sorted, '.', label='Data Points')
# plt.show()

x_weld = y_sorted+115-np.max(y_sorted)
y_weld = -x_sorted+30-np.max(-x_sorted)

# Plot the original data
plt.figure()
plt.plot(x_weld, y_weld, '.', linewidth =0.1, label='Data Points')
plt.xlabel('X Coordinate (mm)')
plt.ylabel('Y Coordinate (mm)')
plt.grid(True)
plt.legend()
plt.show()

# Sort x_weld and y_weld by x_weld
sorted_indices = np.argsort(x_weld)
x_weld_sorted = x_weld[sorted_indices]
y_weld_sorted = y_weld[sorted_indices]

data_to_save = np.column_stack((x_weld_sorted , y_weld_sorted ))
# Save data to a text file with space as delimiter
np.savetxt('weld_coordinates.txt', data_to_save, header='x_weld_sorted  (mm) y_weld_sorted  (mm)', fmt='%.6f')


# Ensure no duplicate x values (optional step depending on your data)
x_weld_unique, unique_indices = np.unique(x_weld_sorted, return_index=True)
y_weld_unique = y_weld_sorted[unique_indices]


# Cubic Spline Interpolation (requires strictly increasing x)
cubic_spline_interp = CubicSpline(x_weld_unique, y_weld_unique)

# Generate more x values for smooth plotting
x_new = np.linspace(x_weld.min(), x_weld.max(), num=1000)
y_cubic = cubic_spline_interp(x_new)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_weld, y_weld, 'o', label='Original Data')
plt.plot(x_new, y_cubic, '--', label='Cubic Spline Interpolation')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolation of Weld Data')
plt.show()

 
# # Assuming x_new and y_cubic are already defined as in previous code
# data_to_save = np.column_stack((x_weld_sorted , y_weld_sorted ))
# # Save data to a text file with space as delimiter
# np.savetxt('r_theta_plot.txt', data_to_save, header='x_weld_sorted  (mm) y_weld_sorted  (mm)', fmt='%.6f')


