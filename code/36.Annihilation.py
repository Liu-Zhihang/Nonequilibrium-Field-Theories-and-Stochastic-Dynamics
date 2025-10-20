import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

def initialize_particles(N0, size=100):
    """Initialize particle positions, ensuring sufficient distance between particles"""
    x = []
    y = []
    
    attempts = 0
    max_attempts = N0 * 100  # Prevent infinite loop
    
    while len(x) < N0 and attempts < max_attempts:
        # Generate candidate position
        new_x = np.random.uniform(0, size)
        new_y = np.random.uniform(0, size)
        
        # Check if it's far enough from existing particles
        valid = True
        min_distance = 3.0  # Minimum distance
        
        for i in range(len(x)):
            distance = np.sqrt((new_x - x[i])**2 + (new_y - y[i])**2)
            if distance < min_distance:
                valid = False
                break
        
        if valid or len(x) == 0:  # First particle is always valid
            x.append(new_x)
            y.append(new_y)
        
        attempts += 1
    
    return np.array(x), np.array(y)

def find_annihilation_pairs(x, y, radius=2.0):
    """Find pairs of particles close enough to annihilate"""
    pairs = []
    used = set()
    
    for i in range(len(x)):
        if i in used:
            continue
        for j in range(i+1, len(x)):
            if j in used:
                continue
            # Calculate distance between particles
            distance = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if distance <= radius:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    
    return pairs

def simulate_annihilation_2d(N0, lambd, t_max, size=100):
    """
    Simulate annihilation process in 2D space
    
    Parameters:
    N0: Initial number of particles
    lambd: Reaction rate constant
    t_max: Maximum simulation time
    size: Simulation area size
    """
    # Initialize particles
    x, y = initialize_particles(N0, size)
    
    # Store trajectory data
    frames = []
    times = [0.0]
    particle_counts = [N0]
    
    t = 0.0
    frames.append((x.copy(), y.copy()))
    
    while t < t_max and len(x) > 1:
        # Calculate reaction propensity
        N = len(x)
        propensity = 0.5 * lambd * N * (N - 1)
        
        if propensity == 0:
            break
            
        # Generate random number to calculate next reaction time
        r1 = np.random.rand()
        dt = (1.0 / propensity) * np.log(1.0 / r1)
        
        # Update time
        t += dt
        
        # Particle random movement (diffusion)
        diffusion_step = np.sqrt(2 * dt) * 0.5  # Slow down diffusion
        x += np.random.normal(0, diffusion_step, len(x))
        y += np.random.normal(0, diffusion_step, len(y))
        
        # Handle boundary conditions (periodic boundaries)
        x = x % size
        y = y % size
        
        # Find and annihilate close particle pairs
        pairs = find_annihilation_pairs(x, y)
        
        # Remove annihilated particles
        if pairs:
            indices_to_remove = []
            for i, j in pairs:
                indices_to_remove.extend([i, j])
            
            # Sort in descending order for easier deletion
            indices_to_remove.sort(reverse=True)
            for idx in indices_to_remove:
                x = np.delete(x, idx)
                y = np.delete(y, idx)
        
        # Save frame data
        times.append(t)
        particle_counts.append(len(x))
        frames.append((x.copy(), y.copy()))
    
    return times, particle_counts, frames, size

# Main program
if __name__ == "__main__":
    # Simulation parameters
    N0 = 100          # Initial particle count
    lambd = 0.001     # Reaction rate
    t_max = 50.0      # Maximum simulation time
    size = 100        # Simulation area size
    
    print("Starting simulation...")
    times, particle_counts, frames, size = simulate_annihilation_2d(N0, lambd, t_max, size)
    print(f"Simulation completed with {len(frames)} frames")
    
    # Create side-by-side animation
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Particle motion animation
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(0, size)
    ax1.set_ylim(0, size)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Annihilation Process: A + A → ∅')
    ax1.grid(True, alpha=0.3)
    
    # Set black background
    ax1.set_facecolor('black')
    fig.set_facecolor('black')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.title.set_color('white')
    
    # Right: Particle count over time
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(0, max(times) if times else t_max)
    ax2.set_ylim(0, N0 + 10)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Particles')
    ax2.set_title('Particle Count Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Set black background
    ax2.set_facecolor('black')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')
    
    # Plot theoretical curve
    t_theory = np.linspace(0, max(times) if times else t_max, 200)
    N_theory = N0 / (1 + N0 * lambd * t_theory)
    ax2.plot(t_theory, N_theory, 'r--', linewidth=2, label='Theory')
    ax2.legend()
    
    # Particle scatter plot (colored particles)
    scat = ax1.scatter([], [], s=50, alpha=0.7)
    
    # Particle count curve
    line2, = ax2.plot([], [], 'b-', linewidth=2, label='Simulation')
    ax2.legend()
    
    # Particle count text
    text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12, 
                   verticalalignment='top', color='white', 
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def animate(frame_idx):
        if frame_idx >= len(frames):
            return [scat, line2, text]
        
        x, y = frames[frame_idx]
        
        # Update particle scatter plot
        points = np.column_stack((x, y)) if len(x) > 0 else np.empty((0, 2))
        scat.set_offsets(points)
        scat.set_color(plt.cm.plasma(np.linspace(0, 1, max(1, len(x)))))
        
        # Update particle count curve
        line2.set_data(times[:frame_idx+1], particle_counts[:frame_idx+1])
        
        # Update text
        text.set_text(f'Time: {times[frame_idx]:.2f}\nParticles: {len(x)}')
        
        return [scat, line2, text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=min(len(frames), 200), interval=200, blit=True, repeat=True)
    
    # Save animation
    print("Saving animation...")
    anim.save('annihilation_colored.gif', writer='pillow', fps=5)
    print("Animation saved as annihilation_colored.gif")
    
    plt.show()