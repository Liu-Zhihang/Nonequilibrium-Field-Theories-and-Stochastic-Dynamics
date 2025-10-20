import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. Define biophysical parameters ---
k_s = 20.0     # Protein synthesis rate (molecules/second)
gamma = 0.2    # Protein degradation rate constant (1/second)

# Theoretical steady-state value
p_ss = k_s / gamma # Steady-state average protein number
noise_strength_N = 2 * k_s # Effective noise strength
sigma = np.sqrt(noise_strength_N) # Volatility in SDE

print(f"Theoretical steady-state mean p_ss = {p_ss:.2f} molecules")
print(f"Effective noise strength N = {noise_strength_N:.2f}")

# --- 2. Simulation parameters ---
p0 = 0.0          # Initial protein number
T_total = 40.0    # Total simulation time (seconds)
dt = 0.05         # Time step
n_steps = int(T_total / dt)
num_cells = 5000  # Number of cells to simulate

# --- 3. Run simulation (Euler-Maruyama) ---
# Initialize protein numbers for all cells
p_paths = np.zeros((num_cells, n_steps + 1))
p_paths[:, 0] = p0

# Generate all random increments
dW = np.sqrt(dt) * np.random.randn(num_cells, n_steps)

# Iteratively solve SDE
for i in range(n_steps):
    current_p = p_paths[:, i]
    drift = k_s - gamma * current_p
    diffusion = sigma * dW[:, i] / dt # Convert to Langevin form noise
    p_paths[:, i+1] = current_p + drift * dt + diffusion * dt

# --- 4. Visualization of results ---
sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, height_ratios=(1, 1))

# Figure 1: Single cell trajectories
ax1 = fig.add_subplot(gs[:, 0])
time_array = np.linspace(0, T_total, n_steps + 1)
for i in range(5): # Plot only 5 trajectories as examples
    ax1.plot(time_array, p_paths[i, :], lw=2, alpha=0.8)

ax1.axhline(p_ss, color='r', linestyle='--', lw=2.5, label=f'Steady-state mean p_ss = {p_ss:.0f}')
ax1.set_title('Stochastic trajectories of protein numbers in single cells', fontsize=18, pad=15)
ax1.set_xlabel('Time (seconds)', fontsize=14)
ax1.set_ylabel('Protein number p(t)', fontsize=14)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_ylim(bottom=0)

# Figure 2: Steady-state distribution
ax2 = fig.add_subplot(gs[0, 1])
final_p_counts = p_paths[:, -1]
sns.histplot(final_p_counts, bins=50, kde=False, stat='density', ax=ax2, 
             color='skyblue', edgecolor='black', label='Simulated distribution (t=40s)')

# Theoretical Gaussian distribution (steady-state solution of O-U process)
variance_theory = noise_strength_N / (2 * gamma)
std_dev_theory = np.sqrt(variance_theory)
p_range = np.linspace(final_p_counts.min(), final_p_counts.max(), 200)
pdf_theory = norm.pdf(p_range, loc=p_ss, scale=std_dev_theory)
ax2.plot(p_range, pdf_theory, 'k-', lw=3, label='Theoretical Gaussian distribution')

ax2.set_title('Protein distribution in cell population at steady-state', fontsize=18, pad=15)
ax2.set_xlabel('Protein number p', fontsize=14)
ax2.set_ylabel('Probability density', fontsize=14)
ax2.legend(fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Figure 3: Ensemble average evolution
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
mean_path = np.mean(p_paths, axis=0)
ax3.plot(time_array, mean_path, color='darkorange', lw=3, label='Ensemble average $\langle p(t) \\rangle$')
ax3.axhline(p_ss, color='r', linestyle='--', lw=2.5)
ax3.set_title('Evolution of ensemble average', fontsize=18, pad=15)
ax3.set_xlabel('Time (seconds)', fontsize=14)
ax3.set_ylabel('Average protein number', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_ylim(bottom=0)

plt.tight_layout()
plt.show()

print(f"\nSimulated steady-state mean: {np.mean(final_p_counts):.2f}")
print(f"Simulated steady-state variance: {np.var(final_p_counts):.2f}")
print(f"Theoretical predicted steady-state variance (N / 2γ): {variance_theory:.2f}")