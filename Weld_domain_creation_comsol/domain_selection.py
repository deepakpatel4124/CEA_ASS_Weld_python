import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import simpledialog, messagebox
from shapely.geometry import Point, Polygon
import matplotlib.patches as patches
import pandas as pd
import os
import pandas as pd
from openpyxl import load_workbook

# Define the number of lines
num_lines = 33

# Define the offset to include small square at the edges
offset = 0.65

# Define the trapezium vertices
A = np.array([82, 30])  # Start point 1
C = np.array([96, 2])   # End point 1
B = np.array([115, 30]) # Start point 2
D = np.array([101, 2])  # End point 2

# Path to the Excel file
file_path = 'Domain_Orientations.xlsx'


# Define the trapezium edges
trapezium = [(A, B), (B, D), (D, C), (C, A)]

# Calculate grid line positions
x_start, x_end = A[0], B[0]
y_start, y_end = C[1], A[1]
width_step = (x_end - x_start) / (num_lines - 1)
height_step = (y_start - y_end) / (num_lines - 1)

# Generate vertical and horizontal lines
vertical_lines = [x_start + i * width_step for i in range(num_lines)]
horizontal_lines = [y_end + i * height_step for i in range(num_lines)]

# Initialize arrays
x_coords_vertical_start = []
y_coords_vertical_start = []
x_coords_vertical_end = []
y_coords_vertical_end = []

x_coords_horizontal_start = []
y_coords_horizontal_start = []
x_coords_horizontal_end = []
y_coords_horizontal_end = []

def intersect_line_with_trapezium(x, trapezium):
    intersections = []
    for (start, end) in trapezium:
        x1, y1 = start
        x2, y2 = end
        if x1 != x2:  # Non-vertical edge
            if min(x1, x2) <= x <= max(x1, x2):
                y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                intersections.append(y)
        else:  # Vertical edge
            if x1 == x:
                intersections.append(min(y1, y2))
                intersections.append(max(y1, y2))
    return sorted(intersections)

def intersect_horizontal_line_with_trapezium(y, trapezium):
    intersections = []
    for (start, end) in trapezium:
        x1, y1 = start
        x2, y2 = end
        if y1 != y2:  # Non-horizontal edge
            if min(y1, y2) <= y <= max(y1, y2):
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(x)
        else:  # Horizontal edge
            if y1 == y:
                intersections.append(min(x1, x2))
                intersections.append(max(x1, x2))
    return sorted(intersections)

# Collect vertical line coordinates
for x in vertical_lines:
    y_coords = intersect_line_with_trapezium(x, trapezium)
    if len(y_coords) >= 2:
        x_coords_vertical_start.append(x)
        y_coords_vertical_start.append(y_coords[0])
        x_coords_vertical_end.append(x)
        y_coords_vertical_end.append(y_coords[1])

# Collect horizontal line coordinates
for y in horizontal_lines:
    x_coords = intersect_horizontal_line_with_trapezium(y, trapezium)
    if len(x_coords) >= 2:
        x_coords_horizontal_start.append(x_coords[0])
        y_coords_horizontal_start.append(y)
        x_coords_horizontal_end.append(x_coords[1])
        y_coords_horizontal_end.append(y)

# Convert to numpy arrays
x_coords_vertical_start = np.array(x_coords_vertical_start)
y_coords_vertical_start = np.array(y_coords_vertical_start)
x_coords_vertical_end = np.array(x_coords_vertical_end)
y_coords_vertical_end = np.array(y_coords_vertical_end)

x_coords_horizontal_start = np.array(x_coords_horizontal_start)
y_coords_horizontal_start = np.array(y_coords_horizontal_start)
x_coords_horizontal_end = np.array(x_coords_horizontal_end)
y_coords_horizontal_end = np.array(y_coords_horizontal_end)

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


# Function to check if a point is inside the trapezium with an offset
def is_point_inside_trapezium(x, y, trapezium, offset):
    # Check if point is inside the trapezium with the given offset
    poly = Polygon([A, B, D, C])
    point = Point(x, y)
    return poly.buffer(offset).contains(point)

# Filter square centers based on being inside the trapezium with offset
filtered_centers = [center for center in square_centers if is_point_inside_trapezium(center[0], center[1], trapezium, offset)]

sorted_centers = sorted(filtered_centers, key=lambda c: (-c[1], c[0]))

# Initialize a list to store square patches
squares = []

def angle_to_color(angle_degrees):
    """
    Convert angle in degrees to a color using a colormap.
    Handles angles in the range [-180, 180] and normalizes to [0, 1].
    """
    # Normalize the angle to the range [0, 360)
    normalized_angle = angle_degrees % 360
    
    # Normalize the angle to the range [0, 1] for colormap
    norm = plt.Normalize(vmin=0, vmax=360)
    
    # Choose a colormap
    cmap = plt.get_cmap('hsv')  # You can choose a different colormap if needed
    
    # Map normalized angle to color
    color = cmap(norm(normalized_angle))
    
    return color

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

class MultiInputDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="X Orientation:").grid(row=0)
        tk.Label(master, text="Y Orientation:").grid(row=1)
        tk.Label(master, text="Z Orientation:").grid(row=2)

        self.x_entry = tk.Entry(master)
        self.y_entry = tk.Entry(master)
        self.z_entry = tk.Entry(master)

        self.x_entry.grid(row=0, column=1)
        self.y_entry.grid(row=1, column=1)
        self.z_entry.grid(row=2, column=1)

        self.x_entry.insert(0, '0')
        self.y_entry.insert(0, '0')
        self.z_entry.insert(0, '0')

        return self.x_entry

    def apply(self):
        self.result = (
            self.x_entry.get(),
            self.y_entry.get(),
            self.z_entry.get()
        )

def ask_orientations(square_no):
    dialog = MultiInputDialog(None, title=f"Orientation Input for Domain {square_no}")
    x_orientation, y_orientation, z_orientation = dialog.result

    try:
        x_orientation = int(x_orientation) if x_orientation else 0
        y_orientation = int(y_orientation) if y_orientation else 0
        z_orientation = int(z_orientation) if z_orientation else 0
    except ValueError:
        x_orientation = y_orientation = z_orientation = 0

    return x_orientation, y_orientation, z_orientation

# Event handler for mouse clicks
def on_click(event):
    if event.inaxes is not None:
        nearest_center, index = find_nearest_center(event.xdata, event.ydata, sorted_centers)
        distance = np.sqrt((nearest_center[0] - event.xdata) ** 2 + (nearest_center[1] - event.ydata) ** 2)
        if distance <= width_step:  # Adjust the threshold as needed
            print(f"Nearest center to clicked point ({event.xdata:.2f}, {event.ydata:.2f}) is center {index + 1} at {nearest_center}")
            plot_square_at_center(ax, nearest_center, width_step, height_step, color='red')  # Change color and size as needed
            fig.canvas.draw()  # Redraw the figure to include the new square

            # Ask for orientation values using the custom dialog
            x_orientation, y_orientation, z_orientation = ask_orientations(index + 1)
            print(f"Orientations - X: {x_orientation}, Y: {y_orientation}, Z: {z_orientation}")
            orientation_data.append([index + 1, nearest_center, x_orientation, y_orientation, z_orientation])  # Store data

            # Plot the square with color based on orientation
            plot_square_at_center(ax, nearest_center, width_step, height_step, color=angle_to_color(y_orientation)) 
            fig.canvas.draw()  # Redraw the figure to include the new square

            # Check if the file already exists
            if os.path.exists(file_path):
                # Read existing data
                existing_df = pd.read_excel(file_path)
                # Remove any rows with the same square number from the existing data
                existing_df = existing_df[~existing_df['Square No.'].isin([data[0] for data in orientation_data])]
                
                # Create a DataFrame from the new data
                new_df = pd.DataFrame(orientation_data, columns=['Square No.', 'Center Coordinates (mm)', 'X Orientation (Degree)', 'Y Orientation (Degree)', 'Z Orientation (Degree)'])
                
                # Combine the filtered existing data with the new data
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Create new DataFrame
                updated_df = pd.DataFrame(orientation_data, columns=['Square No.', 'Center Coordinates (mm)', 'X Orientation (Degree)', 'Y Orientation (Degree)', 'Z Orientation (Degree)'])
            
            # Write updated data to Excel
            updated_df.to_excel(file_path, index=False)
                                                                                                                                                                                                                                     
                                                                                                                      
# Plotting
fig, ax = plt.subplots()

# Plot the trapezium
trapezium_x = [A[0], B[0], D[0], C[0], A[0]]
trapezium_y = [A[1], B[1], D[1], C[1], A[1]]
ax.plot(trapezium_x, trapezium_y, 'k-', label='Weld Zone')

# Plot vertical lines
for i in range(len(x_coords_vertical_start)):
    ax.plot([x_coords_vertical_start[i], x_coords_vertical_end[i]],
             [y_coords_vertical_start[i], y_coords_vertical_end[i]], 'r-',  linewidth = 0.5)

# Plot horizontal lines
for i in range(len(x_coords_horizontal_start)):
    ax.plot([x_coords_horizontal_start[i], x_coords_horizontal_end[i]],
             [y_coords_horizontal_start[i], y_coords_horizontal_end[i]], 'r-', linewidth = 0.5)

# Plot sorted square centers, annotate with numbers, and create square patches
for idx, center in enumerate(sorted_centers):
    x, y = center
    ax.plot(x, y, 'go', markersize=1)  # Adjust markersize as needed
    ax.annotate(str(idx + 1), (x, y), textcoords="offset points", xytext=(0,3), ha='center',fontsize = 8)

if os.path.exists(file_path):
    # Extract data
    df = pd.read_excel(file_path)
    centers = df['Center Coordinates (mm)'].apply(lambda x: eval(x))  # Convert string tuples to actual tuples
    orientations = df['Y Orientation (Degree)']

    # Plot squares from Excel data
    for idx, center in enumerate(centers):
        y_orientation = orientations.iloc[idx]
        color = angle_to_color(y_orientation)
        plot_square_at_center(ax, center, width_step, height_step, color=color)

# Set plot limits
ax.set_xlim([min(trapezium_x) - 1, max(trapezium_x) + 1])
ax.set_ylim([min(trapezium_y) - 1, max(trapezium_y) + 1])

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
root.title("Domain Selection in Weld Zone")

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