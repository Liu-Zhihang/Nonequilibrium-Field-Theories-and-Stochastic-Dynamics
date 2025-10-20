import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon

# --- Parameter settings ---
r = 2.0                # Average event rate
T_max = 10.0             # Total simulation time for trajectories
num_trajectories = 10    # [Modification] Increase the number of displayed trajectories to 10
num_simulations = 20000  # Number of simulations for statistical distribution

# --- Create canvas ---
fig = plt.figure(figsize=(14, 10))
ax_traj = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax_count = plt.subplot2grid((2, 2), (1, 0))
ax_wait = plt.subplot2grid((2, 2), (1, 1))

fig.suptitle(f'Poisson Process: Trajectories and Underlying Distributions (Rate r = {r})', fontsize=16)

# --- 1. Simulate and plot several trajectories (top plot) ---
all_waiting_times = [] 
for i in range(num_trajectories):
    t = 0.0
    x = 0
    times = [t]
    positions = [x]
    
    while t < T_max:
        delta_t = np.random.exponential(scale=1.0/r)
        all_waiting_times.append(delta_t)
        
        t += delta_t
        if t < T_max:
            x += 1
            times.append(t)
            positions.append(x)

    # To make the graph clear, only add labels for the first few trajectories
    if i < 3:
        ax_traj.step(times, positions, where='post', label=f'Sample Trajectory {i+1}')
    else:
        ax_traj.step(times, positions, where='post', alpha=0.6) # Make subsequent trajectories semi-transparent

# Plot theoretical mean line
t_theory = np.linspace(0, T_max, 100)
mean_theory = r * t_theory
ax_traj.plot(t_theory, mean_theory, 'k--', linewidth=2.5, label=f'Theoretical Mean N(t) = {r}*t')
ax_traj.set_title(f'Sample Trajectories of a Poisson Process ({num_trajectories} shown)')
ax_traj.set_xlabel('Time (t)')
ax_traj.set_ylabel('Number of Events (N(t))')
ax_traj.grid(True, linestyle='--', alpha=0.6)
ax_traj.legend()


# --- 2. Plot event count distribution (bottom left plot) ---
event_counts_at_Tmax = np.random.poisson(lam=r * T_max, size=num_simulations)
k_values = np.arange(event_counts_at_Tmax.min(), event_counts_at_Tmax.max() + 1)
ax_count.hist(event_counts_at_Tmax, bins=np.arange(k_values.min(), k_values.max() + 2) - 0.5, density=True, alpha=0.7, label='Simulation')

poisson_pmf_theory = poisson.pmf(k=k_values, mu=r * T_max)
ax_count.plot(k_values, poisson_pmf_theory, 'ro-', label='Theory (Poisson)')
ax_count.set_title(f'Event Count Distribution at T={T_max}')
ax_count.set_xlabel(f'Number of Events (k)')
ax_count.set_ylabel('Probability')
ax_count.set_xticks(k_values[::2])
ax_count.legend()
ax_count.grid(True, linestyle='--', alpha=0.6)


# --- 3. Plot waiting time distribution (bottom right plot) ---
additional_waits = np.random.exponential(scale=1.0/r, size=num_simulations)
all_waiting_times.extend(additional_waits)

ax_wait.hist(all_waiting_times, bins=50, density=True, alpha=0.7, label='Simulation')

t_values_exp = np.linspace(0, max(all_waiting_times), 200)
expon_pdf_theory = expon.pdf(t_values_exp, scale=1.0/r)
ax_wait.plot(t_values_exp, expon_pdf_theory, 'r-', linewidth=2, label='Theory (Exponential)')
ax_wait.set_title('Waiting Time Distribution')
ax_wait.set_xlabel('Time between events (Δt)')
ax_wait.set_ylabel('Probability Density')
ax_wait.legend()
ax_wait.grid(True, linestyle='--', alpha=0.6)


# --- Display final image ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()