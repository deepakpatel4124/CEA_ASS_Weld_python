import matplotlib.pyplot as plt

# Data points for parent (simulation) and weld
parent_LLDL_simu = [(94.25,28.25),(95.00, 15.25)]
parent_LDL_simu = [(93.75,28.25),(93.75,15.5)]
parent_LDLL_simu = [(93.5,28.25),(94.0,13.5)]
parent_LDTT_simu = [(93.25,28.25),(92.75,16.25)]

weld_LLDL_simu = [(94.5,28.25),(94.00, 13.75)]
weld_LDL_simu = [(94.75,28.25),(94.75,16.0)]
weld_LDLL_simu = [(94.5,28.25),(93.5,13.5)]
weld_LDTT_simu = [(93.25,28.25),(92.75,22.75)]

# Data points for experiment (extracted from the image)
parent_LLDL_exp = [(97.5,28.55),(97.5, 13.17)]
parent_LDL_exp = [(97.75,27.56),(97.5,13.16)]
parent_LDLL_exp = [(98.25,28.55),(91.25,14.4)]
parent_LDTT_exp = [(97.25,25.82),(97.5,13.9)]

weld_LLDL_exp = [(98.25,28.55),(97.75, 13.16)]
weld_LDL_exp = [(98.75,27.06),(98.0,15.89)]
weld_LDLL_exp = [(98.25,28.55),(98.5, 13.66)]
weld_LDTT_exp = [(97.75,23.59),(98.25,18.62)]



# Extracting the data points
def plot_data(data, label, marker, color):
    point = data[1]
    line = plt.scatter(point[0], point[1], label=label, marker=marker, color=color, alpha = 0.5)
    line_list.append(line)
    legend1 = plt.legend(handles=line_list, loc='upper right', title="Modes")
    # Add the first legend to the plot
    plt.gca().add_artist(legend1)




line_list =[]
# Plot for parent (simulation) data
# plot_data(parent_LLDL_simu, "Simu LLDL", 'o', 'blue')
plot_data(parent_LDL_simu, "Simu LDL", 's', 'blue')
plot_data(parent_LDLL_simu, "Simu LDLL", 'D', 'blue')
# plot_data(parent_LDTT_simu, "Simu LDTT", '^', 'blue')

# Plot for weld (simulation) data
# plot_data(weld_LLDL_simu, "Simu LLDL", 'o', 'red')
plot_data(weld_LDL_simu, "Simu LDL", 's', 'red')
plot_data(weld_LDLL_simu, "Simu LDLL", 'D', 'red')
# plot_data(weld_LDTT_simu, "Simu LDTT", '^', 'red')

# Set plot labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simulation Data for Parent and Weld")

# Reverse the y-axis
plt.xlim(80,110)
plt.ylim(30,0)

# Second legend (lower left)
exp= plt.scatter([], [], label="Parent",  color='blue')
sim= plt.scatter([], [], label="Weld",  color='red')
plt.legend(handles=[exp, sim], loc='upper left', title="")

plt.grid(True)
plt.show()







line_list =[]
# Plot for parent (experimental) data
plot_data(parent_LLDL_exp, "Exp LLDL", 'o', 'green')
plot_data(parent_LDL_exp, "Exp LDL", 's', 'green')
plot_data(parent_LDLL_exp, "Exp LDLL", 'D', 'green')
plot_data(parent_LDTT_exp, "Exp LDTT", '^', 'green')

# Plot for weld (experimental) data
plot_data(weld_LLDL_exp, "Exp LLDL", 'o', 'orange')
plot_data(weld_LDL_exp, "Exp LDL", 's', 'orange')
plot_data(weld_LDLL_exp, "Exp LDLL", 'D', 'orange')
plot_data(weld_LDTT_exp, "Exp LDTT", '^', 'orange')


# Set plot labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Experiment Data for Parent and Weld")

# Reverse the y-axis
plt.xlim(80,110)
plt.ylim(30,0)

# Second legend (lower left)
exp= plt.scatter([], [], label="Parent",  color='green')
sim= plt.scatter([], [], label="Weld",  color='orange')
plt.legend(handles=[exp, sim], loc='upper left', title="")

# Show the plot
plt.grid(True)
plt.show()

