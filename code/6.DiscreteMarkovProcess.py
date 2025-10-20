import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import matplotlib.patches as mpatches
from collections import Counter

def simulate_markov_process(initial_state, rates, t_max):
    """
    Simulate a discrete state, continuous time Markov process using the Gillespie algorithm.

    Parameters:
        initial_state (int): Initial state of the system (0, 1 or 2).
        rates (dict): Dictionary of transition rates, e.g. {'k_oc': 1.0,...}.
        t_max (float): Maximum simulation time.

    Returns:
        tuple: A tuple containing lists of times and states.
    """
    # Unpack rate parameters
    k_oc = rates.get('k_oc', 0)
    k_co = rates.get('k_co', 0)
    k_ci = rates.get('k_ci', 0)

    # Store trajectory data
    times = [0.0]
    states = [initial_state]

    current_time = 0.0
    current_state = initial_state

    while current_time < t_max:
        # Define possible transitions and their rates in the current state
        possible_transitions = []
        if current_state == 0:  # State O
            possible_transitions = [(1, k_oc)]
        elif current_state == 1:  # State C
            possible_transitions = [(0, k_co), (2, k_ci)]
        elif current_state == 2:  # State I (absorbing state)
            # No outward transitions from state I
            break

        # Calculate total exit rate W
        total_rate = sum(rate for _, rate in possible_transitions)

        if total_rate == 0:
            # Absorbing state, simulation ends
            break

        # Step 1: Determine time interval to next event
        r1 = np.random.rand()
        dt = -np.log(r1) / total_rate
        
        current_time += dt
        if current_time > t_max:
            break

        # Step 2: Determine which event occurs
        r2 = np.random.rand()
        cumulative_prob = 0.0
        next_state = -1
        for state, rate in possible_transitions:
            cumulative_prob += rate / total_rate
            if r2 < cumulative_prob:
                next_state = state
                break

        # Update state and store results
        current_state = next_state
        times.append(current_time)
        states.append(current_state)

    return times, states

def plot_state_diagram(rates):
    """
    Plot state transition diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # State positions
    positions = {0: (2, 4), 1: (2, 2), 2: (2, 0)}  # O, C, I
    state_names = {0: 'Open (O)', 1: 'Closed (C)', 2: 'Inhibited (I)'}
    colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    # Draw state nodes (larger circles)
    for state, pos in positions.items():
        circle = plt.Circle(pos, 0.5, color=colors[state], alpha=0.8)  # Increased radius from 0.3 to 0.5
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state_names[state], ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # Draw transition arrows
    k_oc = rates.get('k_oc', 0)
    k_co = rates.get('k_co', 0)
    k_ci = rates.get('k_ci', 0)
    
    # O <-> C
    if k_oc > 0:
        arrow1 = mpatches.FancyArrowPatch((2, 3.5), (2, 2.5), 
                                          arrowstyle='->', mutation_scale=20, 
                                          color='gray', alpha=0.7, 
                                          connectionstyle="arc3,rad=-0.3")
        ax.add_patch(arrow1)
        ax.text(2.5, 3, f'k_OC = {k_oc}', fontsize=9, rotation=-90, ha='center')
    
    if k_co > 0:
        arrow2 = mpatches.FancyArrowPatch((2, 2.5), (2, 3.5), 
                                          arrowstyle='->', mutation_scale=20, 
                                          color='gray', alpha=0.7,
                                          connectionstyle="arc3,rad=-0.3")
        ax.add_patch(arrow2)
        ax.text(1.5, 3, f'k_CO = {k_co}', fontsize=9, rotation=90, ha='center')
    
    # C -> I
    if k_ci > 0:
        arrow3 = mpatches.FancyArrowPatch((2, 1.5), (2, 0.5), 
                                          arrowstyle='->', mutation_scale=20, 
                                          color='gray', alpha=0.7)
        ax.add_patch(arrow3)
        ax.text(2.5, 1, f'k_CI = {k_ci}', fontsize=9, rotation=-90, ha='center')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('State Transition Diagram', fontsize=14, fontweight='bold')
    
    return fig

def plot_multiple_trajectories(initial_state, rates, t_max, num_trajectories=5):
    """
    Plot multiple trajectories to show randomness
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    state_labels = ['Open (O)', 'Closed (C)', 'Inhibited (I)']
    
    for i in range(num_trajectories):
        times, states = simulate_markov_process(initial_state, rates, t_max)
        # Add small offset to each trajectory for better visualization
        offset_states = [s + (i*0.05) for s in states]
        ax.step(times, offset_states, where='post', alpha=0.7, 
                color=colors[i % len(colors)], linewidth=1.5)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(state_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title(f'Multiple Stochastic Trajectories (N={num_trajectories})')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_state_probability_distribution(initial_state, rates, t_max, num_samples=1000):
    """
    Plot state probability distribution
    """
    # Run multiple simulations to get statistics
    final_states = []
    all_times = []
    all_states = []
    
    for _ in range(num_samples):
        times, states = simulate_markov_process(initial_state, rates, t_max)
        if times:
            final_states.append(states[-1])
            all_times.extend(times)
            all_states.extend(states)
    
    # Count final state distribution
    state_counts = Counter(final_states)
    total = sum(state_counts.values())
    state_probs = {state: count/total for state, count in state_counts.items()}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Final state distribution
    state_labels = ['Open (O)', 'Closed (C)', 'Inhibited (I)']
    state_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    probs = [state_probs.get(i, 0) for i in range(3)]
    
    bars = ax1.bar(state_labels, probs, color=state_colors, alpha=0.7)
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Steady-State Distribution (t={t_max}, N={num_samples})')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2f}', ha='center', va='bottom')
    
    # State occupancy over time (simplified)
    time_points = np.linspace(0, t_max, 50)
    state_occupancy = {0: [], 1: [], 2: []}
    
    for t in time_points:
        count = {0: 0, 1: 0, 2: 0}
        total_count = 0
        for _ in range(100):  # Sample 100 trajectories at each time point
            times, states = simulate_markov_process(initial_state, rates, t_max)
            # Find state at time t
            state_at_t = states[0]  # Default to initial state
            for i in range(len(times)-1):
                if times[i] <= t < times[i+1]:
                    state_at_t = states[i]
                    break
                elif t >= times[-1]:
                    state_at_t = states[-1]
                    break
            count[state_at_t] += 1
            total_count += 1
        
        for state in [0, 1, 2]:
            state_occupancy[state].append(count[state] / total_count if total_count > 0 else 0)
    
    # Plot state occupancy over time
    for state in [0, 1, 2]:
        ax2.plot(time_points, state_occupancy[state], 
                label=state_labels[state], color=state_colors[state], linewidth=2)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Probability')
    ax2.set_title('State Probability Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return fig

# --- Simulation parameters ---
params = {
    'k_oc': 0.5,  # Rate O -> C
    'k_co': 0.2,  # Rate C -> O
    'k_ci': 0.1   # Rate C -> I
}
initial_state = 0  # Start from state O
simulation_time = 100.0

# --- Create enhanced visualizations ---
# 1. State transition diagram
fig1 = plot_state_diagram(params)
plt.show()

# 2. Multiple trajectories
fig2 = plot_multiple_trajectories(initial_state, params, simulation_time, num_trajectories=10)
plt.show()

# 3. State probability distribution
fig3 = plot_state_probability_distribution(initial_state, params, simulation_time, num_samples=500)
plt.show()