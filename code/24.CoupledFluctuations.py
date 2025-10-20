import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, cholesky

# --- 1. Set physical parameters ---
# Define thermodynamic matrix tau (must be symmetric and positive definite)
# tau_11 and tau_22 determine the "cost" of fluctuations of each variable
# tau_12 describes the thermodynamic coupling between the two variables
tau = np.array([[2.0, 0.5],
                [0.5, 1.0]])

# Define kinetic matrix Gamma (can be asymmetric)
# Gamma_11, Gamma_22 are the relaxation rates of each variable
# Gamma_12, Gamma_21 are the kinetic coupling coefficients
Gamma = np.array([[1.0, 0.8],
                  [-0.2, 0.5]])

print("--- System Matrices ---")
print("Thermodynamic Matrix (tau):\n", tau)
print("Kinetic Matrix (Gamma):\n", Gamma)

# --- 2. Derive dependent parameters (apply theory) ---
# Calculate the inverse of tau, which is the equal-time correlation function C(0)
tau_inv = np.linalg.inv(tau)
C0 = tau_inv

# Calculate Onsager matrix L = Gamma * tau_inv
L = Gamma @ tau_inv

# Apply dynamic fluctuation-dissipation theorem to compute noise matrix N = L + L^T
N = L + L.T

print("\n--- Derived Matrices ---")
print("Onsager Matrix (L):\n", L)
print("Noise Covariance Matrix (N):\n", N)

# Check if L satisfies reciprocity relations (here epsilon_a=epsilon_b=1, so L should be symmetric)
# Note: We intentionally chose an asymmetric Gamma, so L is asymmetric here.
# This corresponds to situations where time-reversal symmetry is broken (e.g., by a magnetic field).
# If Gamma were symmetric, L would also be symmetric.

# --- 3. Numerical simulation setup ---
dt = 0.01  # Time step
n_steps = 500000  # Total steps
n_trajectories = 10  # Number of simulated trajectories

# Perform Cholesky decomposition of noise matrix N for generating correlated noise
# N = C * C^T, where C is a lower triangular matrix
try:
    C_noise = cholesky(N, lower=True)
    print("\nCholesky decomposition of N successful.")
except np.linalg.LinAlgError:
    print("\nError: Noise matrix N is not positive definite. Cannot proceed.")
    exit()

# --- 4. Run ensemble simulation ---
print(f"\nRunning simulation for {n_trajectories} trajectories...")
trajectories = np.zeros((n_trajectories, n_steps, 2))

for i in range(n_trajectories):
    phi = np.zeros((n_steps, 2))
    # Sample initial conditions from equilibrium distribution
    # phi[0, :] = cholesky(C0, lower=True) @ np.random.randn(2)
    
    for j in range(1, n_steps):
        # Generate two independent standard normal random numbers
        z = np.random.randn(2)
        # Generate correlated noise through Cholesky factor
        correlated_noise = C_noise @ z
        
        # Euler-Maruyama integration step
        drift = -Gamma @ phi[j-1, :]
        stochastic = correlated_noise / np.sqrt(dt) # Note the dt scaling of the noise term
        
        phi[j, :] = phi[j-1, :] + drift * dt + stochastic * dt
        
    trajectories[i, :, :] = phi

print("Simulation finished.")

# --- 5. Calculate correlation functions ---
def calculate_correlation_function(data, max_lag):
    """Calculate autocorrelation and cross-correlation functions for multiple trajectories"""
    n_traj, n_t, n_var = data.shape
    corr = np.zeros((max_lag, n_var, n_var))
    
    # Use FFT for efficient computation
    for i in range(n_var):
        for j in range(n_var):
            for k in range(n_traj):
                traj_i = data[k, :, i] - np.mean(data[k, :, i])
                traj_j = data[k, :, j] - np.mean(data[k, :, j])
                
                # Calculate cross-correlation
                conv = np.correlate(traj_i, traj_j, mode='full')
                corr[:, i, j] += conv[n_t-1:n_t-1+max_lag]
    
    # Fix broadcasting error: properly normalize for each time delay
    normalization = n_traj * (n_t - np.arange(max_lag))[:, np.newaxis, np.newaxis]
    corr /= normalization
    return corr

max_lag = 500  # Maximum time lag for calculating correlation functions
time_lags = np.arange(max_lag) * dt
C_sim = calculate_correlation_function(trajectories, max_lag)

print("Correlation functions calculated.")

# --- 6. Calculate theoretical predictions and plot ---
C_theory = np.zeros((max_lag, 2, 2))
for i, t in enumerate(time_lags):
    # C(t) = exp(-Gamma * t) * C(0)
    C_theory[i, :, :] = expm(-Gamma * t) @ C0

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
fig.suptitle("Coupled Fluctuations: Simulation vs. Theory", fontsize=16)
labels = [r'$C_{11}(t)$', r'$C_{12}(t)$', r'$C_{21}(t)$', r'$C_{22}(t)$']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i, pos in enumerate(positions):
    ax = axes[pos]
    r, c = pos
    # Simulation results
    ax.plot(time_lags, C_sim[:, r, c], 'o', markersize=4, alpha=0.6, label='Simulation')
    # Theoretical curve
    ax.plot(time_lags, C_theory[:, r, c], 'r-', linewidth=2, label='Theory')
    
    ax.set_title(labels[i])
    ax.set_ylabel('Correlation')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

axes[1, 0].set_xlabel('Time lag t')
axes[1, 1].set_xlabel('Time lag t')

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('Coupled_Fluctuations.png')
plt.show()

# Check the relationship between C12(t) and C21(t)
# Since L is not symmetric, we do not expect C(t) to be symmetric.
# C(t)^T = (expm(-Gamma*t) * C0)^T = C0^T * expm(-Gamma^T*t) = C0 * expm(-Gamma^T*t)
# This is generally not equal to C(t). Our plots also confirm that C12(t) != C21(t).