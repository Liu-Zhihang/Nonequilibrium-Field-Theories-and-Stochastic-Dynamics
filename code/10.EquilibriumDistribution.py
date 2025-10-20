import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from collections import Counter

# Function to simulate a trajectory over time (for the left plot)
def simulate_ehrenfest_trajectory(N=50, n_initial=5, lambda_rate=1.0, t_max=50):
    """
    Simulates a single trajectory of the Ehrenfest model vs. time.
    """
    t = 0.0
    n = n_initial
    times, n_values = [t], [n]
    
    while t < t_max:
        rate_gain = lambda_rate * (N - n)
        rate_lose = lambda_rate * n
        total_rate = rate_gain + rate_lose
        if total_rate == 0: break
        
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        if t > t_max: break
        
        if np.random.rand() < rate_gain / total_rate:
            n += 1
        else:
            n -= 1
            
        times.append(t)
        n_values.append(n)
        
    return times, n_values

# Your function to simulate for the probability distribution (for the right plot)
# This function is efficient and correct, so we keep it as is.
def simulate_for_distribution(N=50, lambda_rate=1.0, num_steps=100000):
    """
    Runs a long simulation (fixed number of jumps) to find the equilibrium distribution.
    """
    n = N // 2
    n_counts = np.zeros(N + 1, dtype=int)
    
    for _ in range(num_steps):
        rate_gain = lambda_rate * (N - n)
        rate_lose = lambda_rate * n
        total_rate = rate_gain + rate_lose
        if total_rate == 0: break
        
        if np.random.rand() < rate_gain / total_rate:
            n += 1
        else:
            n -= 1
        
        n_counts[n] += 1
        
    return n_counts / num_steps

# --- Simulation Parameters ---
N = 50
mean_n = N / 2
std_n = 0.5 * np.sqrt(N)

# --- Create the figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Equilibrium Distribution', fontsize=20, color='darkgreen')

# --- Left Subplot: Stochastic Trajectory ---
times, n_values = simulate_ehrenfest_trajectory(N=N, n_initial=5, t_max=50)
ax1.step(times, n_values, where='post')

ax1.axhline(mean_n, color='orangered', linestyle='--', lw=2)
ax1.axhline(mean_n + std_n, color='green', linestyle='-.', lw=2)
ax1.axhline(mean_n - std_n, color='green', linestyle='-.', lw=2)

ax1.set_xlabel('time')
ax1.set_ylabel('Fleas on Alice')
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 35)
ax1.grid(True)


# --- Right Subplot: Probability Distribution ---
# 1. Run your simulation for the distribution data
simulated_dist = simulate_for_distribution(N=N, num_steps=5000000)

# 2. Calculate the analytical solution
n_range = np.arange(N + 1)
analytical_dist = comb(N, n_range) / (2**N)

# 3. Plot the results, styled to match the PPT
ax2.bar(n_range, simulated_dist, width=1.0, label='simulation', 
        alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
ax2.plot(n_range, analytical_dist, 'ko-', markerfacecolor='black', 
         markeredgecolor='black', markersize=4, lw=1.0, label='analytic solution')

ax2.set_xlabel('Fleas on Alice')
ax2.set_ylabel('Probability')
ax2.set_xlim(12, 38)
ax2.set_ylim(0, 0.125)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)


# --- Final adjustments and display ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()