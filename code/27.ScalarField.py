import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Set simulation parameters ---
grid_size = 100  # Grid size
L = 10.0         # Spatial size (-L/2 to L/2)
dt = 0.01        # Time step
n_steps = 300    # Total steps
omega = 1.0      # Vortex angular velocity

# --- 2. Create grid and field ---
x = np.linspace(-L/2, L/2, grid_size)
y = np.linspace(-L/2, L/2, grid_size)
X, Y = np.meshgrid(x, y)

# Initial scalar field: a Gaussian "dye drop"
c = np.exp(-((X - L/4)**2 + Y**2) / (0.5**2))

# Velocity field: a vortex rotating around the origin
vx = -omega * Y
vy = omega * X

# --- 3. Set up plot ---
fig, ax = plt.subplots(figsize=(7, 6))

# Plot scalar field c
im = ax.imshow(c, extent=[-L/2, L/2, -L/2, L/2], origin='lower',
               cmap='viridis', vmin=0, vmax=1)
fig.colorbar(im, label='Concentration')

# Plot velocity field v (plot every 10 points to keep it clear)
skip = 10
ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
          vx[::skip, ::skip], vy[::skip, ::skip],
          color='white', scale=30)

ax.set_title("Advection of a Scalar Field (t=0.00)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect('equal')

# --- 4. Define animation update function ---
def animate(frame):
    global c
    # Calculate gradient of scalar field
    # np.gradient returns dy, dx order [10, 11]
    grad_c_y, grad_c_x = np.gradient(c, y, x)

    # Calculate advection term v dot grad(c)
    advection_term = vx * grad_c_x + vy * grad_c_y

    # Update scalar field using forward Euler method
    c = c - advection_term * dt

    # Update image data
    im.set_array(c)
    ax.set_title(f"Advection of a Scalar Field (t={frame*dt:.2f})")
    return [im]

# --- 5. Create and run animation ---
ani = FuncAnimation(fig, animate, frames=n_steps,
                    interval=30, blit=True)

# --- 6. Save animation as GIF file ---
# Note: This requires ImageMagick or ffmpeg
# If not installed, you can install with:
# Windows: conda install -c conda-forge imagemagick
# macOS: brew install imagemagick
# Linux: sudo apt-get install imagemagick
ani.save('advection.gif', writer='pillow', dpi=80, fps=20)

plt.show()