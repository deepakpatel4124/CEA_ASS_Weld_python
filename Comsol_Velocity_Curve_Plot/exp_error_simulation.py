import numpy as np
import matplotlib.pyplot as plt

##Parameters##

velocity_correct = 5760 #parent material velocity m/s


x_0_points = np.arange(64)  
y_0_points = [(x, 0) for x in x_0_points]

x_30_points = np.arange(64) 
y_30_points = [(x, 30) for x in x_30_points]

y_0_points, y_30_points

def calculate_distances(x_p, shift_x=0):

    distances_to_y_30 = np.sqrt((x_30_points+shift_x - x_p) ** 2 + 30 ** 2)
    
    return distances_to_y_30

# Example usage with x_p = 10
T_fire = 32
time_ideal = calculate_distances(T_fire)/velocity_correct
time_actual = calculate_distances(T_fire, shift_x=1)/velocity_correct


plt.plot(x_0_points, ((calculate_distances(T_fire, shift_x=1))-calculate_distances(T_fire)), label = 'correct')
plt.legend()
plt.show()

plt.plot(x_0_points, calculate_distances(T_fire)/time_actual, label = 'correct')
plt.axhline(y=5760, color='red', linestyle='--', label='Horizontal Line at 5760')
plt.ylim(5200,6200)
plt.plot








