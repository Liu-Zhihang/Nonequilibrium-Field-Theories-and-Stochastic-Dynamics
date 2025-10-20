import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(x0, mu, sigma, T, dt, num_paths):
    """
    Simulate Geometric Brownian Motion (GBM) using the Euler-Maruyama method.

    Parameters:
    x0 : float
        Initial value.
    mu : float
        Drift rate.
    sigma : float
        Volatility.
    T : float
        Total time.
    dt : float
        Time step.
    num_paths : int
        Number of paths to simulate.

    Returns:
    t : numpy.ndarray
        Array of time points.
    X : numpy.ndarray
        Simulated GBM paths, shape (num_steps, num_paths).
    """
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps + 1)
    
    # Create an array to store paths
    X = np.zeros((num_steps + 1, num_paths))
    X[0, :] = x0
    
    # Generate random numbers for all time steps
    # Z is an array of shape (num_steps, num_paths), each element follows N(0, 1)
    Z = np.random.standard_normal((num_steps, num_paths))
    
    # Iteratively compute each step
    for i in range(num_steps):
        # dW = sqrt(dt) * Z
        dW = np.sqrt(dt) * Z[i]
        # Euler-Maruyama step
        X[i+1] = X[i] + mu * X[i] * dt + sigma * X[i] * dW
        
    return t, X

# --- Simulation parameters ---
x0 = 100.0      # Initial price
mu = 0.05       # Annualized drift (5%)
sigma = 0.2     # Annualized volatility (20%)
T = 1.0         # Total time (1 year)
dt = 0.004      # Time step (approximately one trading day)
num_paths = 50  # Number of paths to simulate

# --- Run simulation ---
t, X = simulate_gbm(x0, mu, sigma, T, dt, num_paths)

# --- Visualization of results ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t, X)
ax.set_title(f'{num_paths} Geometric Brownian Motion Simulation Paths', fontsize=16)
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax.axhline(y=x0, color='r', linestyle='--', label=f'Initial Price = {x0}')
ax.legend()

plt.show()