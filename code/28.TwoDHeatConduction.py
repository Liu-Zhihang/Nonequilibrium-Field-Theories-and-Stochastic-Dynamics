import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
plate_size = 50       # Plate size (NxN grid points)
dx = 0.1              # Spatial step size
D = 4.0               # Thermal diffusivity (mm^2/s)
T_cool = 300.0        # Boundary and initial low temperature
T_hot = 700.0         # Initial hot spot temperature

# Time step must satisfy stability condition
dt = dx**2 / (4 * D)

# --- Initialize Temperature Field ---
u = np.full((plate_size, plate_size), T_cool)

# Set initial hot spot (e.g., a circular region)
radius = 5
for i in range(plate_size):
    for j in range(plate_size):
        if (i-plate_size/2)**2 + (j-plate_size/2)**2 < radius**2:
            u[i, j] = T_hot

# Record temperature field history for animation
u_history = [u.copy()]
num_steps = 200

# --- Time Evolution (Finite Difference Method) ---
def update(u_prev):
    u_new = u_prev.copy()
    # Use NumPy slicing operations to accelerate computation
    u_new[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + D * dt / dx**2 * (
        u_prev[2:, 1:-1] - 2 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1] +
        u_prev[1:-1, 2:] - 2 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]
    )
    # Maintain boundary conditions (fixed temperature at boundaries)
    u_new[0, :] = T_cool
    u_new[-1, :] = T_cool
    u_new[:, 0] = T_cool
    u_new[:, -1] = T_cool
    return u_new

for _ in range(num_steps):
    u = update(u)
    u_history.append(u.copy())

# --- Animation Creation ---
fig, ax = plt.subplots(figsize=(7, 7))

def animate(k):
    ax.clear()
    im = ax.imshow(u_history[k], cmap='hot', vmin=T_cool, vmax=T_hot, interpolation='bilinear')
    ax.set_title(f"Time t = {k * dt:.2f} s")
    # Correctly call set_xticks and set_yticks
    ax.set_xticks([])
    ax.set_yticks([])
    return [im]

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(u_history), interval=50, blit=True)

# Add colorbar
cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
fig.colorbar(ax.imshow(u, cmap='hot', vmin=T_cool, vmax=T_hot), cax=cax, label='Temperature (K)')

# Save the animation as GIF
ani.save('heat_diffusion.gif', writer='pillow', fps=15)

# Show the animation
plt.show()