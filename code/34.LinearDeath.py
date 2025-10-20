import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Use dark background for plots ---
plt.style.use('dark_background')

# --- Simulation Parameters ---
N = 50                 # Initial number of particles
lambda_rate = 0.1      # Per-capita death rate
t_final = 30.0         # Total simulation time
num_trials = 20000     # Number of independent simulations

# --- Run full Gillespie simulations to get all trajectories ---
all_trajectories_n = []
all_trajectories_t = []

# This loop pre-calculates all data needed for the animation
for _ in range(num_trials):
    n, t = N, 0.0
    ns, ts = [n], [t]
    while n > 0:
        total_rate = lambda_rate * n
        # Draw time step from exponential distribution
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        n -= 1 # Only death events
        ns.append(n)
        ts.append(t)
    all_trajectories_n.append(np.array(ns))
    all_trajectories_t.append(np.array(ts))

# --- Create Animation ---
fig, ax = plt.subplots(figsize=(12, 7))
n_values = np.arange(0, N + 1)

# Initialize plot elements
bar_container = ax.bar(n_values, np.zeros(N + 1), width=0.8, label='Stochastic Simulation (Gillespie)', color='cyan', alpha=0.7)
line, = ax.plot(n_values, np.zeros(N + 1), 'ro-', label='Exact Theory (Binomial Distribution)', markersize=5)
time_text = ax.text(0.75, 0.9, '', transform=ax.transAxes, fontsize=14, color='yellow')

def get_pop_at_time(t_point, traj_t, traj_n):
    """Helper function to find the population at a specific time from a trajectory."""
    idx = np.searchsorted(traj_t, t_point, side='right') - 1
    return traj_n[idx]

def update_death_process(frame):
    """Update function for the animation."""
    current_time = frame * t_final / 100
    
    # Get particle counts at the current time from all trajectories
    pops_at_time_t = [get_pop_at_time(current_time, T, n) for T, n in zip(all_trajectories_t, all_trajectories_n)]
    
    # Calculate simulated probability distribution
    counts_sim, _ = np.histogram(pops_at_time_t, bins=np.arange(0, N + 2) - 0.5)
    freq_sim = counts_sim / num_trials
    
    # Update the bar chart heights
    for count, rect in zip(freq_sim, bar_container.patches):
        rect.set_height(count)
    
    # Calculate theoretical probability distribution (Binomial)
    p_survival = np.exp(-lambda_rate * current_time)
    prob_theory = comb(N, n_values) * (p_survival**n_values) * ((1 - p_survival)**(N - n_values))
    
    # Update the theoretical curve
    line.set_ydata(prob_theory)
    
    # Update time text
    time_text.set_text(f'Time t = {current_time:.2f}')
    
    # Dynamically adjust y-axis limit for better visibility
    max_prob = max(np.max(freq_sim), np.max(prob_theory)) if freq_sim.size > 0 else 0.1
    ax.set_ylim(0, max_prob * 1.15)
    
    return bar_container.patches + [line, time_text]

# --- Setup plot aesthetics ---
ax.set_xlabel('Number of Remaining Particles $n$', fontsize=12)
ax.set_ylabel('Probability $P(n, t)$', fontsize=12)
ax.set_title(f'Probability Distribution Evolution for Linear Death Process ($N_0={N}$)', fontsize=16)
ax.legend()
ax.set_xlim(-1, N + 1)

# --- Generate and save the animation ---
ani_death = FuncAnimation(fig, update_death_process, frames=100, interval=100, blit=True)
ani_death.save("linear_death_distribution.gif", writer=PillowWriter(fps=10))
plt.show()