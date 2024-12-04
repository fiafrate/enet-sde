
import matplotlib.pyplot as plt

# Data for different values of ρ
enet_mix_values = [0.1, 0.25, 0.5, 0.75]
rho_0_8_lasso_acc = [0.919674, 0.916769, 0.918736, 0.917185]
rho_0_8_lasso_sel = [0.450937, 0.454062, 0.451719, 0.447734]
rho_0_8_enet_acc = [0.804911, 0.831136, 0.854339, 0.878984]
rho_0_8_enet_sel = [0.845547, 0.804141, 0.752422, 0.660703]

rho_0_5_lasso_acc = [0.957642, 0.95697, 0.95644, 0.95514]
rho_0_5_lasso_sel = [0.904609, 0.905469, 0.903672, 0.899297]
rho_0_5_enet_acc = [0.853019, 0.879099, 0.898139, 0.919794]
rho_0_5_enet_sel = [0.951797, 0.951719, 0.952109, 0.939688]

rho_0_9_lasso_acc = [0.882715, 0.882525, 0.882299, 0.882386]
rho_0_9_lasso_sel = [0.120391, 0.118828, 0.122656, 0.120938]
rho_0_9_enet_acc = [0.764381, 0.791398, 0.81173, 0.838301]
rho_0_9_enet_sel = [0.645312, 0.557813, 0.447891, 0.327266]

# Function to compute averages and append to Enet columns
def append_lasso_averages(lasso_acc, lasso_sel, enet_acc, enet_sel):
    avg_acc = sum(lasso_acc) / len(lasso_acc)
    avg_sel = sum(lasso_sel) / len(lasso_sel)
    enet_acc.append(avg_acc)
    enet_sel.append(avg_sel)
    return enet_acc, enet_sel

# Append averages for each ρ
enet_mix_values.append(1.0)
rho_0_8_enet_acc, rho_0_8_enet_sel = append_lasso_averages(rho_0_8_lasso_acc, rho_0_8_lasso_sel, rho_0_8_enet_acc, rho_0_8_enet_sel)
rho_0_5_enet_acc, rho_0_5_enet_sel = append_lasso_averages(rho_0_5_lasso_acc, rho_0_5_lasso_sel, rho_0_5_enet_acc, rho_0_5_enet_sel)
rho_0_9_enet_acc, rho_0_9_enet_sel = append_lasso_averages(rho_0_9_lasso_acc, rho_0_9_lasso_sel, rho_0_9_enet_acc, rho_0_9_enet_sel)

# Plotting the combined data for all ρ values
plt.figure(figsize=(7, 6))

# Plot for ρ = 0.8
plt.plot(enet_mix_values, rho_0_8_enet_acc, linestyle='-', marker='^', color='blue', label=r'$\rho=0.8$ Accuracy')
plt.plot(enet_mix_values, rho_0_8_enet_sel, linestyle='--', marker='^', color='blue', label=r'$\rho=0.8$ Selection')

# Plot for ρ = 0.5
plt.plot(enet_mix_values, rho_0_5_enet_acc, linestyle='-', marker='d', color='green', label=r'$\rho=0.5$ Accuracy')
plt.plot(enet_mix_values, rho_0_5_enet_sel, linestyle='--', marker='d', color='green', label=r'$\rho=0.5$ Selection')

# Plot for ρ = 0.9
plt.plot(enet_mix_values, rho_0_9_enet_acc, linestyle='-', marker='o', color='red', label=r'$\rho=0.9$ Accuracy')
plt.plot(enet_mix_values, rho_0_9_enet_sel, linestyle='--', marker='o', color='red', label=r'$\rho=0.9$ Selection')

# Adding labels and legend, no title
plt.xlabel(r'Enet Mixing ($\eta$)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('rho-mixing-plot.pdf', format='pdf')
plt.show()