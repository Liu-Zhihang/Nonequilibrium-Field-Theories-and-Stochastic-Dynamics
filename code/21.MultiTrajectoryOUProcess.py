import numpy as np
import matplotlib.pyplot as plt

# --- Parameter definition ---
theta = 0.5  # Reversion rate
mu = 0.0     # Long-term mean
sigma = 0.4  # Noise intensity
dt = 0.01    # Time step
T = 10.0     # Total time
n = int(T / dt) # Total steps
num_paths = 5 # Number of simulated trajectories

# --- Simulation process ---
x = np.zeros((num_paths, n + 1))
x[:, 0] = np.random.uniform(-2, 2, num_paths) # Random initial positions

# Generate all random numbers
Z = np.random.randn(num_paths, n)

for i in range(n):
    # Euler-Maruyama method
    dW = np.sqrt(dt) * Z[:, i]
    dx = -theta * (x[:, i] - mu) * dt + sigma * dW
    x[:, i+1] = x[:, i] + dx

# --- Plotting ---
t = np.linspace(0, T, n + 1)
plt.figure(figsize=(12, 7))

for i in range(num_paths):
    plt.plot(t, x[i, :], lw=1.5, label=f'Path {i+1}')

plt.axhline(mu, color='r', linestyle='--', lw=2, label=f'Mean ($\\mu$={mu})')
plt.title("Ornstein-Uhlenbeck Process Simulation", fontsize=16)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("x(t)", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()