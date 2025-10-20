import numpy as np
import matplotlib.pyplot as plt

def simulate_ehrenfest_trajectory(N=50, n_initial=50, lambda_rate=1.0, t_max=40):
    """
    Simulate a single trajectory of the Ehrenfest model using the Gillespie algorithm.
    """
    t = 0.0
    n = n_initial
    
    times = [t]
    n_values = [n]
    
    while t < t_max:
        rate_gain = lambda_rate * (N - n)
        rate_lose = lambda_rate * n
        total_rate = rate_gain + rate_lose
        
        if total_rate == 0:
            break
            
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        
        if t > t_max:
            break

        if np.random.rand() < rate_gain / total_rate:
            n += 1
        else:
            n -= 1
            
        times.append(t)
        n_values.append(n)
        
    return times, n_values

# --- Simulation parameters ---
N = 50
mean_n = N / 2
std_n = 0.5 * np.sqrt(N)
# Use an appropriate rate to match the trajectory density in the figure
lambda_rate_sim = 1.0

# --- Create figure consistent with PPT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Stochastic Trajectories', fontsize=20, color='darkgreen')

# --- Left subplot: Ladder-like trajectory approaching equilibrium ---
num_trajectories = 4
# Start from a position far from equilibrium (n=5)
n_initial_short = 5 
t_max_short = 50

for i in range(num_trajectories):
    times, n_values = simulate_ehrenfest_trajectory(
        N=N, n_initial=n_initial_short, 
        lambda_rate=lambda_rate_sim, 
        t_max=t_max_short
    )
    # Use step plot to draw ladder-like trajectory
    ax1.step(times, n_values, where='post')

# Plot mean and standard deviation lines
ax1.axhline(mean_n, color='orangered', linestyle='--', lw=2.5, label=r'$\langle n \rangle$')
ax1.axhline(mean_n + std_n, color='green', linestyle='-.', lw=2.5, label=r'$\langle n \rangle \pm \sigma$')
ax1.axhline(mean_n - std_n, color='green', linestyle='-.', lw=2.5)

ax1.set_xlabel('time')
ax1.set_ylabel('Fleas on Alice')
ax1.set_xlim(0, t_max_short)
ax1.set_ylim(0, 38)
# Place legend in the lower right corner
ax1.legend(loc='lower right')
ax1.grid(True)


# --- Right subplot: Long-term equilibrium fluctuations ---
n_initial_long = int(mean_n)
t_max_long = 5000

times_long, n_values_long = simulate_ehrenfest_trajectory(
    N=N, n_initial=n_initial_long, 
    lambda_rate=lambda_rate_sim, 
    t_max=t_max_long
)
# For very dense plots, plot and step look similar, but plot has better performance
ax2.plot(times_long, n_values_long, lw=0.8)

# Plot mean and standard deviation lines
ax2.axhline(mean_n, color='orangered', linestyle='--', lw=2.5)
ax2.axhline(mean_n + std_n, color='green', linestyle='-.', lw=2.5)
ax2.axhline(mean_n - std_n, color='green', linestyle='-.', lw=2.5)

ax2.set_xlabel('time')
ax2.set_xlim(0, t_max_long)
ax2.set_ylim(0, 38)
ax2.grid(True)


# --- Final adjustments and display ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()