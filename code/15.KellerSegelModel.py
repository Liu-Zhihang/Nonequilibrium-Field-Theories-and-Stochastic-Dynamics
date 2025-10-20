import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os

def solve_keller_segel_1d(L=100.0, Nx=200, T=50.0, Nt=5000, D_u=1.0, D_v=20.0, chi_0=25.0, a=1.0):
    """
    Solve the one-dimensional Keller-Segel model using finite difference method.
    u_t = D_u * u_xx - chi_0 * (u * v_x)_x
    v_t = D_v * v_xx + u - a*v
    """
    dx = L / Nx
    dt = T / Nt
    x = np.linspace(0, L, Nx, endpoint=False)
    
    # Check numerical stability conditions (for explicit method)
    # Courant-Friedrichs-Lewy (CFL) condition
    if dt > dx**2 / (2 * max(D_u, D_v)):
        print(f"Warning: Time step {dt:.4f} may be too large for stability.")
        # Adjust time step to ensure stability
        dt = 0.2 * dx**2 / (2 * max(D_u, D_v))
        Nt = int(T / dt)
        print(f"Adjusted time step to {dt:.6f} and Nt to {Nt}")

    # Initialize cell density u and chemical concentration v
    # u: Uniform distribution plus small random perturbation
    u = 1.0 + 0.01 * (np.random.rand(Nx) - 0.5)
    v = u / a  # Initially v is in local equilibrium
    
    u_history = [u.copy()]
    v_history = [v.copy()]
    
    for n in range(Nt):
        # Use periodic boundary conditions
        u_prev = np.roll(u, 1)
        u_next = np.roll(u, -1)
        v_prev = np.roll(v, 1)
        v_next = np.roll(v, -1)
        
        # Calculate second derivatives (Laplacian)
        u_xx = (u_next - 2*u + u_prev) / dx**2
        v_xx = (v_next - 2*v + v_prev) / dx**2
        
        # Calculate first derivatives (Gradient)
        v_x = (v_next - v_prev) / (2*dx)
        
        # Calculate chemotactic flux J = chi * u * v_x
        J = chi_0 * u * v_x
        J_prev = np.roll(J, 1)
        
        # Calculate divergence of flux (Divergence)
        div_J = (J - J_prev) / dx  # Using upwind scheme
        
        # Update u and v
        u_new = u + dt * (D_u * u_xx - div_J)
        v_new = v + dt * (D_v * v_xx + u - a*v)
        
        # Ensure non-negativity of solutions
        u_new = np.maximum(u_new, 0)
        v_new = np.maximum(v_new, 0)
        
        # Check for NaN or infinity
        if np.any(np.isnan(u_new)) or np.any(np.isnan(v_new)) or \
           np.any(np.isinf(u_new)) or np.any(np.isinf(v_new)):
            print(f"Warning: NaN or Inf encountered at step {n}. Stopping simulation.")
            break
            
        u, v = u_new, v_new
        
        if n % 50 == 0:  # Store every 50 steps
            u_history.append(u.copy())
            v_history.append(v.copy())
            
    return x, u_history, v_history

# --- Run simulation ---
# Use parameters more suitable for demonstrating chemotactic collapse
x, u_hist, v_hist = solve_keller_segel_1d(T=100.0, Nt=20000, chi_0=40.0)

# --- Create GIF animation ---
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u_hist[0], 'b-', lw=2)
ax.set_xlim(0, 100)

# Fix y-axis range to better show the chemotactic collapse process
ax.set_ylim(0, 5)  # Set appropriate range based on expected collapse effect
ax.set_xlabel('Position x')
ax.set_ylabel('Cell Density u(x,t)')
ax.set_title('1D Keller-Segel Simulation: Chemotactic Collapse')
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Create temporary directory to store frames
temp_dir = 'temp_frames'
os.makedirs(temp_dir, exist_ok=True)
gif_filename = 'keller_segel_simulation.gif'

def update(frame):
    line.set_ydata(u_hist[frame])
    # Correct time calculation formula
    time = frame * 50 * (100.0/20000)
    time_text.set_text(f'Time = {time:.2f}')
    return line, time_text

# Save each frame as an image
filenames = []
for i in range(len(u_hist)):
    update(i)
    filename = f'{temp_dir}/frame_{i:03d}.png'
    plt.savefig(filename)
    filenames.append(filename)
    if (i+1) % 10 == 0:  # Reduce print frequency
        print(f'Saving frame {i+1}/{len(u_hist)}')

# Use imageio to create GIF, add loop parameter to ensure infinite looping
with imageio.get_writer(gif_filename, mode='I', duration=0.1, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up temporary files
for filename in filenames:
    os.remove(filename)
os.rmdir(temp_dir)

print(f'GIF animation saved as {gif_filename}')

# Plot snapshots at several time points
plt.figure(figsize=(10, 6))
time_snapshots = [0, len(u_hist)//4, len(u_hist)//2, len(u_hist)-1]
for i, frame in enumerate(time_snapshots):
    time = frame * 50 * (100.0/20000)
    plt.plot(x, u_hist[frame], label=f'Time = {time:.2f}')
plt.xlabel('Position x')
plt.ylabel('Cell Density u(x,t)')
plt.title('Snapshots of Cell Density Evolution')
plt.legend()
plt.grid(True)
plt.savefig('keller_segel_snapshots.png', dpi=300, bbox_inches='tight')
plt.show()