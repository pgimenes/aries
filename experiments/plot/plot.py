import matplotlib.pyplot as plt

# x-values representing 'em size'
em_size = [1, 5, 10, 15]

# y-values for 'cot' and 'no cot'
cot_values_sorting32 = [9.5625, 1.93, 1.12, 1.79]
cot_values_sorting64 = [38.83, 16.0, 13.8, 8.87]
no_cot_values_sorting32 = [20.763, 6.05, 12.16, 4.969]


# Create the plot
plt.plot(em_size, cot_values_sorting32, label="Cot-Sorting32", marker='o', linestyle='-', color='blue')
plt.plot(em_size, cot_values_sorting64, label="Cot-Sorting64", marker='o', linestyle='-', color='green')
plt.plot(em_size, no_cot_values_sorting32, label="No Cot-Sorting32", marker='x', linestyle='-', color='red')

# Add labels and title
plt.xlabel('Em Size')
plt.ylabel('Error')
plt.title('Cot vs No Cot on Sorting32 and Sorting64')

# Show legend
plt.legend()

# Display the plot
plt.grid(True)
plt.savefig('cot_vs_no_cot.png')
# plt.show()


# avg number of step per problem for sorting32 with cot
# cot-em1-llm-0-49.log, 15.29
# cot-em5-llm-0-49.log, 14.69
# cot-em10-llm-0-49.log, 12.03
# cot-em15-llm-0-49.log, 12.80

# avg number of step per problem for sorting32 with no cot
# no-cot-em1-llm-0-49.log, 23.22
# no-cot-em5-llm-0-49.log, 22.03
# no-cot-em10-llm-0-49.log, 18.97
# no-cot-em15-llm-0-49.log, 21.72

# x-values representing 'em size'
em_size = [1, 5, 10, 15]

# y-values for avg steps per problem
cot_avg_steps_sorting32 = [15.29, 14.69, 12.03, 12.80]
no_cot_avg_steps_sorting32 = [23.22, 22.03, 18.97, 21.72]

# Create the plot for avg steps
plt.figure()
plt.plot(em_size, cot_avg_steps_sorting32, label="Cot-Sorting32 Avg Steps", marker='o', linestyle='-', color='blue')
plt.plot(em_size, no_cot_avg_steps_sorting32, label="No Cot-Sorting32 Avg Steps", marker='x', linestyle='-', color='red')

# Add labels and title
plt.xlabel('Em Size')
plt.ylabel('Avg Steps')
plt.title('Avg Steps per Problem for Cot vs No Cot on Sorting32')

# Show legend
plt.legend()

# Display the plot
plt.grid(True)
plt.savefig('avg_steps_cot_vs_no_cot.png')
# plt.show()