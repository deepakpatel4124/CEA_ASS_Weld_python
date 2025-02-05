import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import simpledialog, messagebox
from shapely.geometry import Point, Polygon
import matplotlib.patches as patches
from scipy.interpolate import interp1d, CubicSpline
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
from openpyxl import load_workbook


# Number of points to divide the segment into
num_lines_x = 26
num_lines_y = 16

# Define the offset to cover small domain
x_offset = -0.75
y_offset = 1.0




# Path to the Excel file
file_path = 'Orientation_arrow.xlsx'


# Load the data from the text file
weld_curve_data = np.loadtxt('weld_coordinates.txt', delimiter=' ', skiprows=1)
x_weld = weld_curve_data[:, 0]
y_weld = weld_curve_data[:, 1]


# # Flip the x-values around the center
# x_weld = 2 * 90.9 - x_weld

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



# Initialize a list to store square patches
squares = []


# project a set of vectors to the fundamental zone of the cubic crystal
def to_fundamental(data_sol):
    data_sol = np.abs(data_sol)
    data_sol = np.sort(data_sol, axis=-1)
    column = data_sol[...,0].copy()
    data_sol[..., 0] = data_sol[...,1]
    data_sol[..., 1] = column
    return data_sol

# get the color corresponding to a set of vectors
def get_ipf_color(vectors):
    # the following column vectors should map onto R [100], G [010], B[001], i.e. the identity. So the inverse of 
    # this matrix maps the beam directions onto the right color vector
    color_corners = np.array([[0, 1, 1],
                              [0, 0, 1],
                              [1, 1, 1]])
    color_mapper = np.linalg.inv(color_corners)

    # a bit of wrangling
    data_sol = to_fundamental(vectors.data)
    flattened = data_sol.reshape(np.prod(data_sol.shape[:-1]), 3).T
    rgb_mapped = np.dot(color_mapper, flattened)
    rgb_mapped = np.abs(rgb_mapped / rgb_mapped.max(axis=0)).T
    rgb_mapped = rgb_mapped.reshape(data_sol.shape)
    return rgb_mapped

# getting the z vector corresponding to an orientation
def ori_to_vec(eulers):
    from orix.quaternion.rotation import Rotation
    from orix.vector.vector3d import Vector3d
    rotations_regular =  Rotation.from_euler(np.deg2rad(eulers))
    return rotations_regular*Vector3d.zvector()


def angle_to_color(euler_angles):
    """
    Convert a single Z-X-Z Euler angle (with Z and Phi = 0) to an IPF color.
    Assumes that the only rotation is around the X-axis (phi1).
    
    Parameters:
    -A single Euler angle in degrees representing rotation around the X-axis.
    
    Returns:
    - color: A tuple representing the RGB color corresponding to the orientation.
    """
    # Construct the full Euler angle set assuming Phi and phi2 are zero
    euler_angles = np.array([0,euler_angles, 0]).reshape(1, 3)
    
    # Convert Euler angles to a vector in the Z-X-Z rotation sequence
    vector = ori_to_vec(euler_angles)

    # Get the corresponding IPF color for the vector
    ipf_color = get_ipf_color(vector)

    # Return the first (and only) color from the array as a tuple
    return tuple(ipf_color[0])

# Calculate grid line positions
x_start, x_end = np.min(x_weld) , np.max(x_weld)
y_start, y_end = np.min(y_weld) , np.max(y_weld)
width_step = (x_end - x_start) / (num_lines_x - 1)
height_step = (y_start - y_end) / (num_lines_y - 1)
    
# Function to find the nearest center to a given point
def find_nearest_center(x, y, centers):
    distances = [np.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2) for center in centers]
    min_index = np.argmin(distances)
    return centers[min_index], min_index

# Function to plot a square at a given center
def plot_square_at_center(ax, center, width_step,height_step, color):
    
    square = patches.Rectangle((center[0] - width_step/2, center[1] - height_step/2), 
                               width_step, height_step, 
                               linewidth=1, edgecolor='none', facecolor=color, alpha = 0.5)
    ax.add_patch(square)

# Initialize list to store orientation data
orientation_data = []

# Event handler for mouse clicks
def on_click(event):
    if event.inaxes is not None:
        nearest_center, index = find_nearest_center(event.xdata, event.ydata, sorted_centers)
        distance = np.sqrt((nearest_center[0] - event.xdata) ** 2 + (nearest_center[1] - event.ydata) ** 2)
        if distance <= width_step:  # Adjust the threshold as needed
            print(f"Nearest center to clicked point ({event.xdata:.2f}, {event.ydata:.2f}) is center {index + 1} at {nearest_center}")
        
            # Ask for orientation and print it
            orientation = simpledialog.askstring("Orientation Input", f"Orientation of domain {index+1}:")
            if orientation is not None:
                print(f"Orientation of domain: {orientation}")
                for artist in ax.get_children():
                    if isinstance(artist, patches.Rectangle):  # Assuming squares are rectangles
                        if artist.get_xy() == (nearest_center[0] - width_step/2, nearest_center[1] - height_step/2):
                            artist.remove()
              
                orientation_data.append([index + 1, nearest_center, int(orientation)])  # Store data
                plot_square_at_center(ax, nearest_center, width_step, height_step, color=angle_to_color(int(orientation)))  # Change color and size as needed
                fig.canvas.draw()  # Redraw the figure to include the new square
            
                # Check if the file already exists
                if os.path.exists(file_path):
                    # Read existing data
                    existing_df = pd.read_excel(file_path)
                    # Remove any rows with the same square number from the existing data
                    existing_df = existing_df[~existing_df['Square No.'].isin([data[0] for data in orientation_data])]
                    
                    # Create a DataFrame from the new data
                    new_df = pd.DataFrame(orientation_data, columns=['Square No.', 'Center Coordinates (mm)', 'Orientation (Degree)'])
                    
                    # Combine the filtered existing data with the new data
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                    # Create new DataFrame
                    updated_df = pd.DataFrame(orientation_data, columns=['Square No.', 'Center Coordinates (mm)', 'Orientation (Degree)'])
                
                # Write updated data to Excel                                                                                                                             
                updated_df.to_excel(file_path, index=False) 
                                                                                                                                                                                                                                     
                                                                                                                      
# Plotting
fig, ax = plt.subplots()

x_new = np.linspace(x_weld.min(), x_weld.max(), num=100)
y_cubic = cubic_spline_interp(x_new)

# Plot the weld profile
ax.plot(x_new, y_cubic, '-', linewidth =1, label='Data Points')

# Plot vertical lines
for i in range(len(x_coords_vertical_start)):
    ax.plot([x_coords_vertical_start[i], x_coords_vertical_end[i]],
             [y_coords_vertical_start[i], y_coords_vertical_end[i]], 'r-')

# Plot horizontal lines
for i in range(len(x_coords_horizontal_start)):
    ax.plot([x_coords_horizontal_start[i], x_coords_horizontal_end[i]],
             [y_coords_horizontal_start[i], y_coords_horizontal_end[i]], 'b-')

# Plot sorted square centers, annotate with numbers, and create square patches
for idx, center in enumerate(sorted_centers):
    x, y = center
    ax.plot(x, y, 'go', markersize=1)  # Adjust markersize as needed
    ax.annotate(str(idx + 1), (x, y), textcoords="offset points", xytext=(0,3), ha='center',fontsize = 8)

def plot_arrow(ax, center, orientation, length):
    # Convert orientation in degrees to radians
    angle_rad = np.radians(orientation)
    
    # Calculate the end point of the arrow
    x, y = center
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    
    # Add an arrow to the plot
    ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
if os.path.exists(file_path):
    # Extract data
    df = pd.read_excel(file_path)
    centers = df['Center Coordinates (mm)'].apply(lambda x: eval(x))  # Convert string tuples to actual tuples
    orientations = df['Orientation (Degree)']

    # Plot squares from Excel data
    for idx, center in enumerate(centers):
        orientation = orientations.iloc[idx]
        color = angle_to_color(orientation)
        plot_square_at_center(ax, center, width_step, height_step, color=color)
        
# Set plot limits
ax.set_xlim([np.min(x_weld) - 1, np.max(x_weld) + 1])
ax.set_ylim([np.min(y_weld) - 1, np.max(y_weld) + 1])

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

# Connect the event handler
fig.canvas.mpl_connect('button_press_event', on_click)

# Create the Tkinter window
root = tk.Tk()
root.geometry("1600x1200")
root.title("Weld with Domains")

# Embed the Matplotlib figure into the Tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Add the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Start the Tkinter main loop
root.mainloop()

print(len(filtered_centers))