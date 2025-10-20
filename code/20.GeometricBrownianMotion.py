import numpy as np
import matplotlib.pyplot as plt
# Can use sdeint library for more complex SDE solving
# import sdeint

# --- Parameter settings ---
x0 = 1.0          # Initial value
T = 1.0           # Total time
N = 1000          # Number of time steps
dt = T / N        # Time step size
mu = 0.1          # Drift rate
sigma = 0.4       # Volatility
num_paths = 500   # Number of simulated trajectories

# --- Simulation ---
# Generate Wiener increments shared by all paths
dW = np.random.normal(0, np.sqrt(dt), (num_paths, N))

# Initialize path arrays
x_ito = np.zeros((num_paths, N + 1))
x_stratonovich = np.zeros((num_paths, N + 1))
x_ito[:, 0] = x0
x_stratonovich[:, 0] = x0

# Iterative solving
for i in range(N):
    # Ito (Euler-Maruyama)
    x_ito[:, i+1] = x_ito[:, i] + mu * x_ito[:, i] * dt + sigma * x_ito[:, i] * dW[:, i]
    
    # Stratonovich (Euler-Heun)
    # Prediction step
    x_pred = x_stratonovich[:, i] + mu * x_stratonovich[:, i] * dt + sigma * x_stratonovich[:, i] * dW[:, i]
    # Correction step
    g_avg = 0.5 * (sigma * x_stratonovich[:, i] + sigma * x_pred)
    x_stratonovich[:, i+1] = x_stratonovich[:, i] + mu * x_stratonovich[:, i] * dt + g_avg * dW[:, i]

# --- Plotting ---
time_axis = np.linspace(0, T, N + 1)
mean_ito = np.mean(x_ito, axis=0)
mean_stratonovich = np.mean(x_stratonovich, axis=0)

# Theoretical mean
# Ito interpretation: E[x(t)] = x0 * exp(mu * t)
# Stratonovich interpretation: E[x(t)] = x0 * exp((mu + 0.5*sigma^2) * t)
theoretical_mean_ito = x0 * np.exp(mu * time_axis)
theoretical_mean_stratonovich = x0 * np.exp((mu + 0.5 * sigma**2) * time_axis)

plt.figure(figsize=(10, 6))
plt.plot(time_axis, mean_ito, label='Ito (Euler-Maruyama) Mean')
plt.plot(time_axis, mean_stratonovich, label='Stratonovich (Euler-Heun) Mean')
plt.plot(time_axis, theoretical_mean_ito, 'k--', label='Theoretical Ito Mean $x_0 e^{\mu t}$')
# plt.plot(time_axis, theoretical_mean_stratonovich, 'g:', label='Theoretical Stratonovich Mean')
plt.title('Comparison of Ito and Stratonovich Simulations (Mean of 500 paths)')
plt.xlabel('Time $t$')
plt.ylabel('Mean value $\langle x(t) \\rangle$')
plt.legend()
plt.grid(True)
plt.show()