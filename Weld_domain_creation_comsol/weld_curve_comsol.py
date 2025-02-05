import mph
import numpy as np
import pickle
import sys
from shapely.geometry import Point, Polygon
from scipy.interpolate import interp1d, CubicSpline

# Start COMSOL client
client = mph.start()

# Load the COMSOL model
pymodel = client.load('201_Domain_26_08\Weld_Curve_2.mph')
model = pymodel.java

print("Running...")



# Number of points to divide the segment into
num_lines_x = 26
num_lines_y = 16

# Define the offset to cover small domain
x_offset = -0.75
y_offset = 1.0


# Load the data from the text file
weld_curve_data = np.loadtxt('weld_coordinates.txt', delimiter=' ', skiprows=1)
x_weld = weld_curve_data[:, 0]
y_weld = weld_curve_data[:, 1]

# Ensure no duplicate x values (optional step depending on your data)
x_weld_unique, unique_indices = np.unique(x_weld, return_index=True)
y_weld_unique = y_weld[unique_indices]

# Cubic Spline Interpolation (requires strictly increasing x)
cubic_spline_interp = CubicSpline(x_weld_unique, y_weld_unique)

# Function to find the roots of the difference between interpolated y and target y_value
def find_x_for_y(cubic_spline, y_value, x_range):
    # Define the function whose root we want to find (difference from target y_value)
    def root_function(x):
        return cubic_spline(x) - y_value

    # Find the roots (there may be multiple x values for the same y)
    from scipy.optimize import root_scalar

    x_values = []
    for i in range(len(x_range) - 1):
        try:
            result = root_scalar(root_function, bracket=[x_range[i], x_range[i+1]])
            if result.converged:
                x_values.append(result.root)
        except ValueError:
            pass  # Skip intervals where no root is found

    return x_values


# Divide the line segment AB into num_lines parts
x_AB = np.linspace(np.min(x_weld), np.max(x_weld), num_lines_x)


# Calculate y-coordinates on the u curve for the x-coordinates of points on AB
y_curve_v = cubic_spline_interp(x_AB)


# Divide y-values of the segment AC
y_AC = np.linspace(np.min(y_weld), np.max(y_weld), num_lines_y)   

# Initialize lists to store results
x_curve_H = []
x_curve_H_alt  = []

for y_val in y_AC:
    # Generate x range for searching the roots
    x_range = np.linspace(x_weld.min(), x_weld.max(), num=100)
    
    # Find x values corresponding to the given y value
    x_for_y = find_x_for_y(cubic_spline_interp, y_val, x_range)
    
    # Ensure there are at least two x values for each y_val
    if len(x_for_y) == 2:
        x_curve_H.append((x_for_y[0]))
        x_curve_H_alt.append(x_for_y[1])
    else:
        # Handle cases where fewer than 2 x values are found (optional)
        x_curve_H.append(np.nan)  # or handle as needed
        x_curve_H_alt.append(np.nan)  # or handle as needed

# Convert lists to numpy arrays for easy handling
x_curve_H = np.array(x_curve_H)
x_curve_H_alt = np.array(x_curve_H_alt)


# Initialize arrays
x_coords_vertical_start = []
y_coords_vertical_start = []
x_coords_vertical_end = []
y_coords_vertical_end = []

x_coords_horizontal_start = []
y_coords_horizontal_start = []
x_coords_horizontal_end = []
y_coords_horizontal_end = []

# Populate arrays for vertical and horizontal lines
for x, y in zip(x_AB, y_curve_v):
    x_coords_vertical_start.append(x)
    y_coords_vertical_start.append(np.max(y_weld))
    x_coords_vertical_end.append(x)
    y_coords_vertical_end.append(y)

for x1, x2, y in zip(x_curve_H, x_curve_H_alt, y_AC):
    x_coords_horizontal_start.append(x1)
    y_coords_horizontal_start.append(y)
    x_coords_horizontal_end.append(x2)
    y_coords_horizontal_end.append(y)


# Convert to numpy arrays
x_coords_vertical_start = np.round(np.array(x_coords_vertical_start),1)
y_coords_vertical_start = np.round(np.array(y_coords_vertical_start),1)
x_coords_vertical_end = np.round(np.array(x_coords_vertical_end),1)
y_coords_vertical_end = np.round(np.array(y_coords_vertical_end),1)

x_coords_horizontal_start = np.round(np.array(x_coords_horizontal_start),1)
y_coords_horizontal_start = np.round(np.array(y_coords_horizontal_start),1)
x_coords_horizontal_end = np.round(np.array(x_coords_horizontal_end),1)
y_coords_horizontal_end = np.round(np.array(y_coords_horizontal_end),1)


def is_point_below_curve(x, y):
    if x>(np.mean(x_weld)):
        x = x + x_offset
    else:
        x = x- x_offset
    return y + y_offset>= cubic_spline_interp(x)

# Find the centers of the squares
square_centers = []

for i in range(len(x_coords_horizontal_start) - 1):
    for j in range(len(x_coords_vertical_start) - 1):
        x1 = x_coords_vertical_start[j]
        x2 = x_coords_vertical_start[j + 1]
        y1 = y_coords_horizontal_start[i]
        y2 = y_coords_horizontal_start[i + 1]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        square_centers.append((center_x, center_y))
        

# Check if the centers are inside the curve
filtered_centers = [(x, y) for (x, y) in square_centers if is_point_below_curve(x, y)]

sorted_centers = sorted(filtered_centers, key=lambda c: (-c[1], c[0]))


for i in range(0, num_lines_x):
    
    ls = f"ls{130+i}"
    
    model.component("comp1").geom("geom1").create(ls, "LineSegment");
    
    model.component("comp1").geom("geom1").feature(ls).set("specify1", "coord");
    model.component("comp1").geom("geom1").feature(ls).set("specify2", "coord");

    model.component("comp1").geom("geom1").feature(ls).set("coord1", [x_coords_vertical_start[i], y_coords_vertical_start[i]]);
    model.component("comp1").geom("geom1").feature(ls).set("coord2", [x_coords_vertical_end[i], y_coords_vertical_end[i]]);
    
for i in range(0, num_lines_y):
    
    ls = f"ls{130+num_lines_x+i}"
    
    model.component("comp1").geom("geom1").create(ls, "LineSegment");
    
    model.component("comp1").geom("geom1").feature(ls).set("specify1", "coord");
    model.component("comp1").geom("geom1").feature(ls).set("specify2", "coord");

    model.component("comp1").geom("geom1").feature(ls).set("coord1", [x_coords_horizontal_start[i], y_coords_horizontal_start[i]]);
    model.component("comp1").geom("geom1").feature(ls).set("coord2", [x_coords_horizontal_end[i], y_coords_horizontal_end[i]]);
    
    
model.component("comp1").geom("geom1").run();

for i in range(0, len(sorted_centers)):
    disksel = f"disksel{i+1}"
    model.component("comp1").geom("geom1").create(disksel, "DiskSelection");
    model.component("comp1").geom("geom1").feature(disksel).set("posx", sorted_centers[i][0]);
    model.component("comp1").geom("geom1").feature(disksel).set("posy", sorted_centers[i][1]);
    model.component("comp1").geom("geom1").feature(disksel).set("r", 1.8);
    model.component("comp1").geom("geom1").feature(disksel).set("condition", "inside");



# Save the model with the updated interpolation function
model.save('201_Domain_26_08\Weld_Curve_3.mph')

print("Finished...")

