import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameter settings (same as before) ---
k = 2.0; gamma = 1.0; T = 1.0; beta = 1.0 / T
dt = 0.01; t_f = 5.0; num_steps = int(t_f / dt)
num_trajectories = 20000 # Increase trajectory count for smoother distribution

# --- 2. Forward protocol simulation ---
lambda_0_fwd = 0.0; lambda_f_fwd = 5.0
v_lambda_fwd = (lambda_f_fwd - lambda_0_fwd) / t_f
work_forward = np.zeros(num_trajectories)

print("Simulating forward protocol...")
for i in range(num_trajectories):
    x = np.random.normal(loc=lambda_0_fwd, scale=np.sqrt(T / k))
    total_work = 0.0
    for step in range(num_steps):
        t = step * dt
        lambda_t = lambda_0_fwd + v_lambda_fwd * t
        force_lambda = k * (x - lambda_t)
        # Key correction: definition of work
        dW = -force_lambda * (v_lambda_fwd * dt)
        total_work += dW
        force_x = -k * (x - lambda_t)
        noise_term = np.sqrt(2 * gamma * T * dt) * np.random.randn()
        x += (force_x / gamma) * dt + noise_term / gamma
    work_forward[i] = total_work

# --- 3. Reverse protocol simulation ---
lambda_0_rev = lambda_f_fwd; lambda_f_rev = lambda_0_fwd
v_lambda_rev = (lambda_f_rev - lambda_0_rev) / t_f # Velocity is negative
work_reverse = np.zeros(num_trajectories)

print("Simulating reverse protocol...")
for i in range(num_trajectories):
    x = np.random.normal(loc=lambda_0_rev, scale=np.sqrt(T / k))
    total_work = 0.0
    for step in range(num_steps):
        t = step * dt
        lambda_t = lambda_0_rev + v_lambda_rev * t
        force_lambda = k * (x - lambda_t)
        # Key correction: definition of work
        dW = -force_lambda * (v_lambda_rev * dt)
        total_work += dW
        force_x = -k * (x - lambda_t)
        noise_term = np.sqrt(2 * gamma * T * dt) * np.random.randn()
        x += (force_x / gamma) * dt + noise_term / gamma
    work_reverse[i] = total_work

# --- 4. Visualizing Crooks Theorem ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Numerical Verification of Crooks Fluctuation Theorem', fontsize=18)

# --- Figure 1: Intersection point of work distributions ---
delta_F = 0.0
bins = np.linspace(-15, 40, 75)
ax1.hist(work_forward, bins=bins, density=True, alpha=0.7, label=r'Forward work distribution $P(W)$')
# Plot distribution of negative reverse work P_R(-W)
ax1.hist(-work_reverse, bins=bins, density=True, alpha=0.7, label=r'Reverse work distribution $P_R(-W)$')
ax1.axvline(delta_F, color='red', linestyle='--', linewidth=2, label=f'ΔF = {delta_F:.1f}')
ax1.set_xlabel('Work W', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Verification Method 1: Work Distributions Intersect at ΔF', fontsize=14)
ax1.legend()

# --- Figure 2: Linear relationship of logarithmic probability ratio ---
# Calculate histogram data
hist_fwd, bin_edges = np.histogram(work_forward, bins=bins, density=True)
hist_rev, _ = np.histogram(-work_reverse, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Avoid division by zero, only select regions where both distributions are non-zero
mask = (hist_fwd > 1e-4) & (hist_rev > 1e-4)
log_ratio = np.log(hist_fwd[mask] / hist_rev[mask])
work_vals = bin_centers[mask]

# Plot scatter diagram
ax2.scatter(work_vals, log_ratio, alpha=0.8, label='Simulation data (ln[P(W)/P_R(-W)])')

# Plot theoretical prediction line
w_theory = np.linspace(np.min(work_vals), np.max(work_vals), 100)
log_ratio_theory = beta * (w_theory - delta_F)
ax2.plot(w_theory, log_ratio_theory, 'r-', lw=3, label=f'Theoretical prediction: Slope β={beta:.1f}')

ax2.set_xlabel('Work W', fontsize=12)
ax2.set_ylabel(r'$\ln[P(W) / P_R(-W)]$', fontsize=14)
ax2.set_title('Verification Method 2: Linear Relationship of Log Probability Ratio', fontsize=14)
ax2.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()