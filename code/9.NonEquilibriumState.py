import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as patches

def gillespie_simulation(Q, initial_state, T_max):
    """
    Simulates a continuous-time Markov process using the Gillespie algorithm.
    
    Args:
        Q (np.array): The transition rate matrix (Q_ij = w_j->i).
        initial_state (int): The starting state.
        T_max (float): The maximum simulation time.
        
    Returns:
        tuple: A tuple containing lists of times, states, and transitions.
    """
    num_states = Q.shape[0]
    states = np.arange(num_states)
    
    current_state = initial_state
    t = 0.0
    
    time_points = [t]
    state_trajectory = [current_state]
    transitions = []

    while t < T_max:
        rates_out = np.array([Q[j, current_state] for j in states if j != current_state])
        total_rate_out = np.sum(rates_out)
        
        if total_rate_out == 0:
            break

        dt = np.random.exponential(1.0 / total_rate_out)
        t += dt
        if t >= T_max:
            break
            
        possible_next_states = [j for j in states if j != current_state]
        probabilities = rates_out / total_rate_out
        next_state = np.random.choice(possible_next_states, p=probabilities)
        
        transitions.append((current_state, next_state))
        current_state = next_state
        
        time_points.append(t)
        state_trajectory.append(current_state)
        
    return time_points, state_trajectory, transitions

# --- 2. Irreversible System: Driven Particle on a Ring ---
print("--- Simulating Irreversible System (Non-Equilibrium Steady State) ---")
Q_irrev = np.zeros((3, 3))

# Define rates that model a strong clockwise driving force
w_cw = 10.0  # High rate for clockwise jumps
w_ccw = 0.5   # Low rate for counter-clockwise jumps

w01_i, w12_i, w20_i = w_cw, w_cw, w_cw
w10_i, w21_i, w02_i = w_ccw, w_ccw, w_ccw

# Populate Q-matrix
Q_irrev[1, 0] = w01_i; Q_irrev[0, 1] = w10_i
Q_irrev[2, 1] = w12_i; Q_irrev[1, 2] = w21_i
Q_irrev[0, 2] = w20_i; Q_irrev[2, 0] = w02_i

for i in range(3):
    Q_irrev[i, i] = -np.sum(Q_irrev[:, i])
    
# Verify violation of Kolmogorov's criterion
prod_clockwise_i = w01_i * w12_i * w20_i
prod_counter_clockwise_i = w02_i * w21_i * w10_i
print("Kolmogorov's Loop Criterion Check:")
print(f"  Clockwise rate product = {prod_clockwise_i:.4f}")
print(f"  Counter-clockwise rate product = {prod_counter_clockwise_i:.4f}")
print("The products are unequal, confirming the system is irreversible.\n")

# Run simulation
T_max_irrev = 5000
times_irrev, states_irrev, transitions_irrev = gillespie_simulation(Q_irrev, 0, T_max_irrev)

# Count jumps to show net flux
transition_counts_i = Counter(transitions_irrev)
clockwise_jumps_i = transition_counts_i.get((0,1),0) + transition_counts_i.get((1,2),0) + transition_counts_i.get((2,0),0)
counter_clockwise_jumps_i = transition_counts_i.get((0,2),0) + transition_counts_i.get((2,1),0) + transition_counts_i.get((1,0),0)

print(f"In a simulation of T={T_max_irrev}:")
print(f"  Total clockwise jumps: {clockwise_jumps_i}")
print(f"  Total counter-clockwise jumps: {counter_clockwise_jumps_i}")
print("A strong net clockwise probability current is observed.\n")

# Calculate and plot cumulative entropy production
rates_map = {
    (0,1): w01_i, (1,0): w10_i, 
    (1,2): w12_i, (2,1): w21_i, 
    (2,0): w20_i, (0,2): w02_i
}
entropy_production = 0.0
cumulative_entropy = [0.0]
entropy_times = [0.0]

for i in range(len(transitions_irrev)):
    start, end = transitions_irrev[i]
    ds_env = np.log(rates_map[(start, end)] / rates_map[(end, start)])
    entropy_production += ds_env
    cumulative_entropy.append(entropy_production)
    entropy_times.append(times_irrev[i+1])

# Calculate average entropy production rate, sigma
avg_sigma = cumulative_entropy[-1] / entropy_times[-1]
print(f"Simulated average entropy production rate (sigma) ~ {avg_sigma:.4f}")

# --- Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

# Plot 1: System Diagram
ax1 = fig.add_subplot(gs[0])
pos = {0: (0, 1), 1: (0.866, -0.5), 2: (-0.866, -0.5)}
ax1.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=1000, c=['#FF6B6B', '#4ECDC4', '#45B7D1'])
labels = ['Site A (0)', 'Site B (1)', 'Site C (2)']
for i, p in pos.items():
    ax1.text(p[0], p[1], labels[i], ha='center', va='center', color='white', fontweight='bold')

# Arrows for rates
# Clockwise (strong)
ax1.add_patch(patches.FancyArrowPatch(pos[0], pos[1], connectionstyle="arc3,rad=.3", color="black", arrowstyle='->,head_length=10,head_width=8', linewidth=2.5))
ax1.add_patch(patches.FancyArrowPatch(pos[1], pos[2], connectionstyle="arc3,rad=.3", color="black", arrowstyle='->,head_length=10,head_width=8', linewidth=2.5))
ax1.add_patch(patches.FancyArrowPatch(pos[2], pos[0], connectionstyle="arc3,rad=.3", color="black", arrowstyle='->,head_length=10,head_width=8', linewidth=2.5))
# Counter-clockwise (weak)
ax1.add_patch(patches.FancyArrowPatch(pos[1], pos[0], connectionstyle="arc3,rad=.3", color="gray", arrowstyle='->,head_length=5,head_width=4', linewidth=1, linestyle='--'))
ax1.add_patch(patches.FancyArrowPatch(pos[2], pos[1], connectionstyle="arc3,rad=.3", color="gray", arrowstyle='->,head_length=5,head_width=4', linewidth=1, linestyle='--'))
ax1.add_patch(patches.FancyArrowPatch(pos[0], pos[2], connectionstyle="arc3,rad=.3", color="gray", arrowstyle='->,head_length=5,head_width=4', linewidth=1, linestyle='--'))

ax1.text(0, 1.5, 'Driving Force', ha='center', fontsize=12, style='italic', color='darkred')
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 2.0)
ax1.set_aspect('equal'); ax1.axis('off')
ax1.set_title('Physical Model: Driven Particle on a Ring')


# Plot 2: Cumulative Entropy Production
ax2 = fig.add_subplot(gs[1])
ax2.plot(entropy_times, cumulative_entropy, '-', label=r'Cumulative Entropy Production $\Delta s(t)$')
ax2.plot(entropy_times, avg_sigma * np.array(entropy_times), 'k--', linewidth=2, label=fr'Average Rate $\sigma \approx {avg_sigma:.3f}$')
ax2.set_title('Entropy Production in the Driven System')
ax2.set_xlabel('Time')
ax2.set_ylabel(r'Total Entropy Production, $\Delta s(t)$')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()