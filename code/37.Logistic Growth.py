import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameter Settings ---
n0 = 10.0         # Initial population size
mu = 0.5          # Birth rate
lambda_rate = 0.01 # Competition rate
T_max = 100.0     # Total simulation time
dt = 0.01         # Time step
n_steps = int(T_max / dt)
n_traj = 5        # Number of stochastic trajectories

# Carrying capacity K
K = 2 * mu / lambda_rate

# --- 2. Time Axis ---
t = np.linspace(0, T_max, n_steps + 1)

# --- 3. Simulate Stochastic Trajectories ---
plt.figure(figsize=(12, 7))
# Set black background
plt.style.use('dark_background')

for j in range(n_traj):
    n_stochastic = np.zeros(n_steps + 1)
    n_stochastic[0] = n0
    for i in range(n_steps):
        n = n_stochastic[i]
        if n <= 0:
            n_stochastic[i+1:] = 0
            break
        
        # Drift term V(n)
        drift = mu * n - (lambda_rate / 2) * n * (n - 1)
        
        # Diffusion term D(n)
        diffusion_term_squared = mu * n + (lambda_rate / 2) * n * (n - 1)
        
        # Noise amplitude sqrt(2D(n))
        noise_amp = np.sqrt(diffusion_term_squared) if diffusion_term_squared > 0 else 0
        
        # Euler-Maruyama step
        noise = np.random.normal(0, 1)
        n_stochastic[i+1] = n + drift * dt + noise_amp * np.sqrt(dt) * noise
        
    plt.plot(t, n_stochastic, lw=1.5, alpha=0.8, label=f'Stochastic Trajectory {j+1}')

# --- 4. Calculate Deterministic (Mean Field) Solution ---
# Solution to dn/dt = μn(1 - n/K)
n_deterministic = K / (1 + (K/n0 - 1) * np.exp(-mu * t))
plt.plot(t, n_deterministic, 'r-', lw=2.5, label='Deterministic Logistic Eq.')

# --- 5. Plotting ---
plt.axhline(K, color='cyan', linestyle=':', lw=2, label=f'Carrying Capacity K = {K:.0f}')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Population Size (n)', fontsize=14)
plt.title('Stochastic vs. Deterministic Logistic Growth', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.savefig('LogisticGrowth.png')
plt.show()

