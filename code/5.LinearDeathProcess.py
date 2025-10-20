import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# --- Simulation parameters ---
n0 = 20      # Initial population size
lambda_rate = 1.0  # Per capita death rate
t_max = 4.0    # Maximum simulation time
num_simulations = 10000 # Total number of simulated trajectories

# --- Gillespie algorithm implementation for linear death process ---
def linear_death_gillespie(n_start, rate, t_end):
    """
    Simulate linear death process using Gillespie algorithm
    """
    t = 0.0
    n = n_start
    
    times = [t]
    populations = [n]
    
    while n > 0 and t < t_end:
        # Total reaction rate
        alpha = rate * n
        
        # Generate waiting time tau
        xi1 = np.random.uniform(0, 1)
        tau = -np.log(xi1) / alpha
        
        # Update time and population
        t += tau
        n -= 1 # Only death events
        
        times.append(t)
        populations.append(n)
        
    return np.array(times), np.array(populations)

# --- Run multiple simulations ---
all_trajectories = []
for _ in range(num_simulations):
    times, populations = linear_death_gillespie(n0, lambda_rate, t_max)
    all_trajectories.append((times, populations))

# --- Plotting ---
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Figure 1: Stochastic trajectories and deterministic solution ---
ax = axes[0]
# Plot some example trajectories
for i in range(3):
    times, populations = all_trajectories[i]
    ax.step(times, populations, where='post')

# Plot deterministic solution
t_deterministic = np.linspace(0, t_max, 200)
n_deterministic = n0 * np.exp(-lambda_rate * t_deterministic)
ax.plot(t_deterministic, n_deterministic, color='black', lw=2, label=r'$n(t) = n_0 e^{-\lambda t}$')

ax.set_xlabel('t', fontsize=14)
ax.set_ylabel('population size', fontsize=14)
ax.set_title('stochastic trajectories', fontsize=16)
ax.set_ylim(0, n0 + 2)
ax.set_xlim(0, t_max - 0.5)

# --- Figure 2: Histogram and theoretical solution at different times ---
ax = axes[1]
time_points = [0.1, 1.0, 4.0]
colors = ['darkgreen', 'saddlebrown', 'darkred']
bar_width = 0.8

# Helper function: get population size at a specific time from a trajectory
def get_population_at_time(trajectory, t_point):
    times, populations = trajectory
    # Find the index of the last time point <= t_point
    idx = np.searchsorted(times, t_point, side='right') - 1
    return populations[idx]

for t, color in zip(time_points, colors):
    # Collect population sizes at this time point from all simulations
    populations_at_t = [get_population_at_time(traj, t) for traj in all_trajectories]
    
    # Calculate histogram
    n_values, counts = np.unique(populations_at_t, return_counts=True)
    probabilities = counts / num_simulations
    
    # Plot histogram
    ax.bar(n_values, probabilities, width=bar_width, color=color, alpha=0.7, ec='black', label=f't = {t}')
    
    # Plot theoretical binomial distribution
    p_survival = np.exp(-lambda_rate * t)
    n_range = np.arange(0, n0 + 1)
    p_n_t = comb(n0, n_range) * (p_survival**n_range) * ((1 - p_survival)**(n0 - n_range))
    # For visualization, only plot points with significant probabilities
    # ax.plot(n_range, p_n_t, 'o', color=color, markersize=4)

ax.set_xlabel('n', fontsize=14)
ax.set_ylabel('p(n, t)', fontsize=14)
ax.set_title('histogram', fontsize=16)
ax.set_xlim(-1, n0 + 1)
ax.set_ylim(0, 0.7)

# Add theoretical formula text
ax.text(10, 0.5, r'$p(n,t) = \binom{n_0}{n} p^n q^{n_0-n}$', fontsize=14, color='saddlebrown')
ax.text(1.5, 0.65, 't = 4', fontsize=12, color='darkred')
ax.text(7, 0.2, 't = 1', fontsize=12, color='saddlebrown')
ax.text(17, 0.35, 't = 0.1', fontsize=12, color='darkgreen')

fig.suptitle('Linear Death Process', fontsize=20, color='darkgreen', y=1.02)
plt.tight_layout()
plt.show()