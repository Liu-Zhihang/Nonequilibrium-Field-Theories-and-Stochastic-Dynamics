import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
L = 2.0                 # Domain size
nx, ny = 100, 100       # Grid resolution
nu = 0.005              # Kinematic viscosity (higher for visible effect)
dt = 0.005              # Time step
n_steps = 100           # Number of time steps
n_steps_per_frame = 5   # Steps between frames

# --- Grid Setup ---
x = np.linspace(-L/2, L/2, nx)
y = np.linspace(-L/2, L/2, ny)
X, Y = np.meshgrid(x, y)
dx = L / (nx - 1)

# --- Initial Velocity Field (Rankine Vortex) ---
# Create a more realistic vortex with both rotational core and potential flow outer region
r = np.sqrt(X**2 + Y**2)
R_core = 0.3            # Core radius

# Velocity components for a Rankine vortex
# Inside core (solid body rotation)
inside = r <= R_core
u = np.zeros_like(X)
v = np.zeros_like(Y)

# Inside the core: solid body rotation (v = omega * r)
u[inside] =  Y[inside] * 2.0 / R_core**2
v[inside] = -X[inside] * 2.0 / R_core**2

# Outside the core: potential vortex (v = Gamma / (2*pi*r))
outside = r > R_core
u[outside] =  Y[outside] / (2 * np.pi * r[outside]**2)
v[outside] = -X[outside] / (2 * np.pi * r[outside]**2)

# Scale the vortex
u *= 0.5
v *= 0.5

# --- Vorticity Calculation ---
def calculate_vorticity(u, v, dx):
    """Calculate vorticity field from velocity field"""
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dx)
    dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx)
    vorticity = dvdx - dudy
    return vorticity

# --- Numerical Solver (Navier-Stokes with viscous terms) ---
def viscous_step(u, v, nu, dt, dx):
    """Solve Navier-Stokes equations with viscous terms"""
    # Temporary arrays
    u_new = u.copy()
    v_new = v.copy()
    
    # Update interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # Second derivatives (Laplacians)
            d2u_dx2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2
            d2u_dy2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
            d2v_dx2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dx**2
            d2v_dy2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx**2
            
            # First derivatives
            du_dx = (u[i, j+1] - u[i, j-1]) / (2*dx)
            du_dy = (u[i+1, j] - u[i-1, j]) / (2*dx)
            dv_dx = (v[i, j+1] - v[i, j-1]) / (2*dx)
            dv_dy = (v[i+1, j] - v[i-1, j]) / (2*dx)
            
            # Nonlinear terms (advection)
            DuDt_conv = -u[i, j]*du_dx - v[i, j]*du_dy
            DvDt_conv = -u[i, j]*dv_dx - v[i, j]*dv_dy
            
            # Diffusion terms (viscous)
            DuDt_diff = nu * (d2u_dx2 + d2u_dy2)
            DvDt_diff = nu * (d2v_dx2 + d2v_dy2)
            
            # Update velocities
            u_new[i, j] = u[i, j] + dt * (DuDt_conv + DuDt_diff)
            v_new[i, j] = v[i, j] + dt * (DvDt_conv + DvDt_diff)
    
    # Boundary conditions (zero velocity at boundaries)
    u_new[0, :] = 0.0
    u_new[-1, :] = 0.0
    u_new[:, 0] = 0.0
    u_new[:, -1] = 0.0
    v_new[0, :] = 0.0
    v_new[-1, :] = 0.0
    v_new[:, 0] = 0.0
    v_new[:, -1] = 0.0
    
    return u_new, v_new

# --- Visualization Setup ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: Velocity field with streamlines
ax1.set_xlim(-L/2, L/2)
ax1.set_ylim(-L/2, L/2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Velocity Field and Streamlines')
ax1.set_aspect('equal')

# Stream plot for velocity field
speed = np.sqrt(u**2 + v**2)
strm = ax1.streamplot(X, Y, u, v, color=speed, cmap='viridis', density=1.5)
fig.colorbar(strm.lines, ax=ax1, label='Speed')

# Right subplot: Vorticity field
ax2.set_xlim(-L/2, L/2)
ax2.set_ylim(-L/2, L/2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Vorticity Field')
ax2.set_aspect('equal')

# Calculate initial vorticity
initial_vorticity = np.zeros_like(X)
vorticity_interior = calculate_vorticity(u, v, dx)
initial_vorticity[1:-1, 1:-1] = vorticity_interior

# Vorticity visualization
im = ax2.imshow(initial_vorticity, extent=[-L/2, L/2, -L/2, L/2], 
                origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
fig.colorbar(im, ax=ax2, label='Vorticity')

# Time text
time_text = fig.text(0.02, 0.95, '', fontsize=12,
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

# Energy tracking
energies = []

# --- Animation Function ---
def animate(frame):
    global u, v
    
    # Perform several time steps between frames for smoother animation
    kinetic_energy = 0
    for _ in range(n_steps_per_frame):
        u, v = viscous_step(u, v, nu, dt, dx)
        # Calculate kinetic energy
        kinetic_energy += 0.5 * np.sum(u**2 + v**2) * dx**2
    
    energies.append(kinetic_energy)
    
    # Update stream plot (clear and redraw)
    for artist in ax1.collections + ax1.lines:
        artist.remove()
    speed = np.sqrt(u**2 + v**2)
    strm = ax1.streamplot(X, Y, u, v, color=speed, cmap='viridis', density=1.5)
    
    # Update vorticity plot
    vorticity = np.zeros_like(X)
    vorticity_interior = calculate_vorticity(u, v, dx)
    vorticity[1:-1, 1:-1] = vorticity_interior
    im.set_array(vorticity)
    
    # Update time text
    current_time = frame * n_steps_per_frame * dt
    time_text.set_text(f'Time = {current_time:.3f}')
    
    return strm.lines, im

# --- Create Animation ---
ani = animation.FuncAnimation(
    fig, animate, frames=n_steps//n_steps_per_frame, 
    interval=100, blit=False, repeat=True)

plt.tight_layout()
#plt.show()

# To save the animation:
ani.save('vortex_decay_detailed.gif', writer='pillow', fps=10)