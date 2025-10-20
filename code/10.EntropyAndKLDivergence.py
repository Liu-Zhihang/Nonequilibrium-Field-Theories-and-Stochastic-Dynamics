import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def simulate_ensemble(N=30, n_initial=30, num_trajectories=5000, t_points=np.linspace(0, 15, 100)):
    """
    Simulates an ensemble of Ehrenfest trajectories to calculate p_n(t).
    
    Args:
        N (int): Total number of fleas.
        n_initial (int): Initial state for all trajectories.
        num_trajectories (int): Number of trajectories in the ensemble.
        t_points (numpy.ndarray): Time points at which to calculate p_n(t).
        
    Returns:
        numpy.ndarray: A 2D array p_n_t where p_n_t[i, j] is p_j(t_i).
    """
    # Using a simpler simulation method for ensemble (tau-leaping approximation) for speed
    # This is sufficient for demonstrating the concept.
    # A full Gillespie ensemble would be more accurate but much slower.
    
    p_n_t = np.zeros((len(t_points), N + 1))
    p_n_t[0, n_initial] = 1.0  # At t=0, all systems are in n_initial
    
    dt = t_points[1] - t_points[0] if len(t_points) > 1 else 0
    lambda_rate = 0.1  # This rate affects the time scale
    
    for i in range(1, len(t_points)):
        p_prev = p_n_t[i-1, :]
        p_curr = np.copy(p_prev)
        
        # Evolution of probabilities based on the Master Equation
        for n in range(N + 1):
            # Flow out of state n
            p_curr[n] -= p_prev[n] * lambda_rate * (n + (N - n)) * dt
            # Flow into state n
            if n > 0:
                p_curr[n] += p_prev[n-1] * lambda_rate * (N - (n - 1)) * dt
            if n < N:
                p_curr[n] += p_prev[n+1] * lambda_rate * (n + 1) * dt
        
        p_n_t[i, :] = p_curr
        
    return p_n_t

# --- Calculation and Plotting ---
N = 30
t_eval = np.linspace(0, 15, 150)
p_n_t = simulate_ensemble(N=N, n_initial=N, t_points=t_eval)

# Calculate S(t) and H(t)
pi_n = comb(N, np.arange(N + 1)) / (2**N)
log_comb_N_n = np.log(comb(N, np.arange(N + 1)))

S_t = []
H_t = []

for p_n in p_n_t:
    # Avoid log(0) issues
    p_n_safe = p_n[p_n > 0]
    pi_n_safe = pi_n[p_n > 0]
    
    # S(t) calculation
    term1 = -np.sum(p_n_safe * np.log(p_n_safe))
    term2 = np.sum(p_n * log_comb_N_n)
    S_t.append(term1 + term2)
    
    # H(t) calculation
    H = np.sum(p_n_safe * np.log(p_n_safe / pi_n_safe))
    H_t.append(H)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(t_eval, S_t)
ax1.set_title('Statistical Entropy S(t)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Entropy S(t)')
ax1.grid(True)

ax2.plot(t_eval, H_t)
ax2.set_title('Kullback-Leibler Divergence H(t)')
ax2.set_xlabel('Time')
ax2.set_ylabel('KL Divergence H(t)')
ax2.grid(True)

plt.tight_layout()
plt.show()