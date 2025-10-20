import numpy as np
import matplotlib.pyplot as plt

def random_walk_1d_two_boundaries(start_pos, n_steps, boundary_left=0, boundary_right=10):
    """
    Simulate a one-dimensional random walk with two absorbing boundaries.
    
    Parameters:
    start_pos (int): Initial position
    n_steps (int): Maximum number of steps
    boundary_left (int): Position of the left absorbing boundary
    boundary_right (int): Position of the right absorbing boundary
    
    Returns:
    list: List of positions in the walk trajectory
    """
    position = start_pos
    path = [position]
    
    for _ in range(n_steps):
        # Check if either absorbing boundary has been reached
        if position == boundary_left or position == boundary_right:
            # Absorbing boundary reached, stay in place until simulation ends
            path.append(position)
            continue
            
        # Equal probability to move left or right by one step
        step = np.random.choice([-1, 1])
        position += step
        path.append(position)
        
    return path

# --- Simulation parameters ---
initial_position = 5
max_steps = 500
num_trajectories = 20
boundary_left_pos = 0
boundary_right_pos = 20

# --- Run and plot multiple trajectories ---
plt.figure(figsize=(12, 7))

for i in range(num_trajectories):
    path = random_walk_1d_two_boundaries(
        initial_position, 
        max_steps, 
        boundary_left_pos, 
        boundary_right_pos
    )
    # To make trajectories clearer, give different trajectories some color variation
    plt.plot(path, alpha=0.7, color=plt.cm.cool(i / num_trajectories))

# Plot absorbing boundaries
plt.axhline(y=boundary_left_pos, color='r', linestyle='--', linewidth=2, label=f'Left Absorbing Boundary (Position {boundary_left_pos})')
plt.axhline(y=boundary_right_pos, color='r', linestyle='--', linewidth=2, label=f'Right Absorbing Boundary (Position {boundary_right_pos})')

plt.title(f'{num_trajectories} Random Walk Trajectories in [{boundary_left_pos}, {boundary_right_pos}] Interval')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
# Adjust y-axis range to better display boundaries
plt.ylim(bottom=boundary_left_pos - 1, top=boundary_right_pos + 1) 

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

plt.savefig('1D_random_walk_two_boundaries.jpg', dpi=300)
plt.show()
plt.close()