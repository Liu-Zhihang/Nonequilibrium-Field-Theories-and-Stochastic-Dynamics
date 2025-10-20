import numpy as np
import matplotlib.pyplot as plt

# --- Parameter settings ---
T = 10.0          # Total time
N = 1000          # Number of time steps
dt = T / N        # Time step size
D = 0.5           # Diffusion coefficient (2D = 1)
num_paths = 500   # Number of simulated trajectories

# --- Simulate Wiener process ---
# Create an array of shape (num_paths, N+1) to store all trajectories
# N+1 because it includes the initial point at t=0
paths = np.zeros((num_paths, N + 1))

# Generate random increments for all time steps
# np.random.normal(loc=mean, scale=std_dev, size=...)
# std_dev = sqrt(variance) = sqrt(2*D*dt)
increments = np.random.normal(0, np.sqrt(2 * D * dt), (num_paths, N))

# Build trajectories by accumulating increments
# paths[:, 1:] means filling from the second time point onwards
# np.cumsum(..., axis=1) performs cumulative sum along the time axis (columns)
paths[:, 1:] = np.cumsum(increments, axis=1)

# --- Plotting ---
# 1. Plot several sample trajectories
plt.figure(figsize=(14, 6))
time_axis = np.linspace(0, T, N + 1)

plt.subplot(1, 2, 1)
for i in range(min(num_paths, 10)): # Only plot first 10 to maintain clarity
    plt.plot(time_axis, paths[i, :], lw=1)
plt.title(f'{min(num_paths, 10)} Sample Wiener Process Paths')
plt.xlabel('Time $t$')
plt.ylabel('$W(t)$')
plt.grid(True)

# 2. Verify mean squared displacement
# Calculate mean squared displacement at each time point
# np.mean(paths**2, axis=0) computes mean along the trajectory axis (rows)
mean_squared_displacement = np.mean(paths**2, axis=0)
theoretical_msd = 2 * D * time_axis

plt.subplot(1, 2, 2)
plt.plot(time_axis, mean_squared_displacement, label='Simulated $\\langle W^2(t) \\rangle$')
plt.plot(time_axis, theoretical_msd, 'r--', label='Theoretical $2Dt$')
plt.title('Mean Squared Displacement')
plt.xlabel('Time $t$')
plt.ylabel('$\\langle W^2(t) \\rangle$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()