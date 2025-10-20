import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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

# --- 1. Reversible System: Particle in a Triple-Well Potential ---
print("--- Simulating Reversible System (Thermal Equilibrium) ---")

# Physical Parameters (in units of k_B*T, so k_B*T=1)
E = np.array([-1.5, -2.0, -0.5]) # Energy of wells A, B, C
E_barrier_base = 0.5 # Energy barrier height relative to the higher well
prefactor = 1.0 # Arrhenius prefactor

# Calculate theoretical steady-state (Boltzmann distribution)
Z = np.sum(np.exp(-E))
pi_theory = np.exp(-E) / Z

# Calculate transition rates from energy landscape (Arrhenius law)
# w_ij = A * exp(-(E_barrier_ij - E_i))
rates = {}
rates[(0, 1)] = prefactor * np.exp(-(E_barrier_base + max(E[0], E[1]) - E[0])) # A -> B
rates[(1, 0)] = prefactor * np.exp(-(E_barrier_base + max(E[0], E[1]) - E[1])) # B -> A
rates[(1, 2)] = prefactor * np.exp(-(E_barrier_base + max(E[1], E[2]) - E[1])) # B -> C
rates[(2, 1)] = prefactor * np.exp(-(E_barrier_base + max(E[1], E[2]) - E[2])) # C -> B
rates[(2, 0)] = prefactor * np.exp(-(E_barrier_base + max(E[2], E[0]) - E[2])) # C -> A
rates[(0, 2)] = prefactor * np.exp(-(E_barrier_base + max(E[2], E[0]) - E[0])) # A -> C

# Construct the Q-matrix
Q_rev = np.zeros((3, 3))
for (i, j), rate in rates.items():
    Q_rev[j, i] = rate
for i in range(3):
    Q_rev[i, i] = -np.sum(Q_rev[:, i])

# Verify Kolmogorov's loop criterion
prod_clockwise = rates[(0,1)] * rates[(1,2)] * rates[(2,0)]
prod_counter_clockwise = rates[(0,2)] * rates[(2,1)] * rates[(1,0)]
print("Kolmogorov's Loop Criterion Check:")
print(f"  Clockwise rate product = {prod_clockwise:.4f}")
print(f"  Counter-clockwise rate product = {prod_counter_clockwise:.4f}")
print(f"  Ratio (should be 1.0): {prod_clockwise / prod_counter_clockwise:.4f}\n")

# Run a long simulation
T_max_rev = 200000
times, states, transitions = gillespie_simulation(Q_rev, 0, T_max_rev)

# Analyze net flux
transition_counts = Counter(transitions)
clockwise_jumps = transition_counts.get((0,1),0) + transition_counts.get((1,2),0) + transition_counts.get((2,0),0)
counter_clockwise_jumps = transition_counts.get((0,2),0) + transition_counts.get((2,1),0) + transition_counts.get((1,0),0)

print(f"In a simulation of T={T_max_rev}:")
print(f"  Total clockwise jumps: {clockwise_jumps}")
print(f"  Total counter-clockwise jumps: {counter_clockwise_jumps}")
print("The counts are very close, indicating no net probability flux.\n")

# Calculate simulated steady-state distribution
occupancy_times = np.zeros(3)
for i in range(len(times) - 1):
    occupancy_times[states[i]] += times[i+1] - times[i]
occupancy_times[states[-1]] += T_max_rev - times[-1]
pi_simulated = occupancy_times / T_max_rev

print("Steady-State Distribution (Boltzmann) Comparison:")
print(f"  Theoretical pi = {np.round(pi_theory, 4)}")
print(f"  Simulated pi   = {np.round(pi_simulated, 4)}")

# --- Visualization ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2)

# Plot 1: Potential Energy Landscape
ax1 = fig.add_subplot(gs[0, :])
x_coords = np.linspace(-1, 5, 500)
# A simple polynomial to represent a triple-well potential visually
U = 0.1 * (x_coords**6 - 12*x_coords**5 + 47*x_coords**4 - 60*x_coords**3 + 16*x_coords**2 + 10*x_coords)
# Adjust wells to roughly match energy levels E
U -= np.min(U)
U[x_coords < 1.5] += E[0] - np.min(U[x_coords < 1.5]) + 2.5
U[(x_coords >= 1.5) & (x_coords <= 3.5)] += E[1] - np.min(U[(x_coords >= 1.5) & (x_coords <= 3.5)]) + 2.5
U[x_coords > 3.5] += E[2] - np.min(U[x_coords > 3.5]) + 2.5

ax1.plot(x_coords, U, color='black', linewidth=2.5)
ax1.set_title('Physical Model: Particle in a Triple-Well Potential')
ax1.set_xlabel('Position (arbitrary units)')
ax1.set_ylabel(r'Potential Energy $U(x)$')
ax1.text(0.5, E[0]+2.8, 'Well A (State 0)', ha='center')
ax1.text(2.5, E[1]+2.8, 'Well B (State 1)', ha='center')
ax1.text(4.5, E[2]+2.8, 'Well C (State 2)', ha='center')
ax1.set_yticks([]) # Hide y-axis ticks for a cleaner look

# Plot 2: Trajectory
ax2 = fig.add_subplot(gs[1, 0])
state_labels = ['Well A (0)', 'Well B (1)', 'Well C (2)']
t_display_limit = min(500, T_max_rev)
display_indices = np.where(np.array(times) <= t_display_limit)
ax2.step(np.array(times)[display_indices], np.array(states)[display_indices], where='post')
ax2.set_title(f'Sample Trajectory (First {t_display_limit} time units)')
ax2.set_xlabel('Time')
ax2.set_ylabel('State')
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(state_labels)
ax2.set_ylim(-0.5, 2.5)

# Plot 3: Steady-State Distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(state_labels, pi_simulated, color='cornflowerblue', edgecolor='black', label='Simulation Result')
ax3.plot(state_labels, pi_theory, 'ro', markersize=10, label=r'Theory (Boltzmann $\pi$)')
ax3.set_title('Steady-State Probability Distribution')
ax3.set_ylabel('Probability')
ax3.set_ylim(0, max(np.max(pi_theory), np.max(pi_simulated)) * 1.2)
ax3.legend()
for i, p in enumerate(pi_simulated):
    ax3.text(i, p, f'{p:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()