import matplotlib.pyplot as plt

# x-values representing 'em size'
em_size = [1, 5, 10, 15]

# y-values for 'cot' and 'no cot'
cot_values = [9.5625, 1.93, 1.12, 1.79]
no_cot_values = [20.763, 6.05, 12.16, 4.969]

# Create the plot
plt.plot(em_size, cot_values, label="Cot", marker='o', linestyle='-', color='blue')
plt.plot([1, 5, 10, 15], no_cot_values, label="No Cot", marker='x', linestyle='-', color='red')

# Add labels and title
plt.xlabel('Em Size')
plt.ylabel('Value')
plt.title('Cot vs No Cot')

# Show legend
plt.legend()

# Display the plot
plt.grid(True)
plt.savefig('cot_vs_no_cot.png')
# plt.show()
