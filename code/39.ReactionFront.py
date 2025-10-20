import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import matplotlib.animation as animation

# --- 1. Simulation Parameters ---
GRID_WIDTH = 256
GRID_HEIGHT = 128
D = 0.2              # Diffusion coefficient
MU = 0.1             # Reaction rate (birth rate)
K = 1.0              # Carrying capacity
T_FINAL = 300.0      # Total simulation time
DT = 0.1             # Time step
N_STEPS = int(T_FINAL / DT)
DX = 1.0             # Spatial step

# --- 2. Initial Condition: A front on the left side ---
n0 = np.zeros((GRID_HEIGHT, GRID_WIDTH))
front_width = 10
n0[:, :front_width] = K

# --- 3. Setup for Numerical Simulation ---
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / (DX**2)
n_deterministic = n0.copy()
n_stochastic = n0.copy()

# Store history for animation
history_det = [n_deterministic.copy()]
history_sto = [n_stochastic.copy()]
save_interval = 20 # Save one frame every 20 steps

# --- 4. Main Simulation Loop ---
print("Running simulations for reaction fronts...")
for step in range(N_STEPS):
    # --- Deterministic (Mean-Field) Simulation ---
    lap_det = convolve2d(n_deterministic, laplacian_kernel, mode='same', boundary='wrap')
    reaction_det = MU * n_deterministic * (1 - n_deterministic / K)
    n_deterministic += (D * lap_det + reaction_det) * DT

    # --- Stochastic (KMPI) Simulation ---
    n_stochastic[n_stochastic < 0] = 0
    
    # Drift part (same as deterministic)
    lap_sto = convolve2d(n_stochastic, laplacian_kernel, mode='same', boundary='wrap')
    reaction_sto = MU * n_stochastic * (1 - n_stochastic / K)
    drift_term = D * lap_sto + reaction_sto
    
    # Noise part: Combined reaction and diffusion noise
    # The term is approximately sqrt(2*D*n + mu*n)
    noise_strength_sq = 2 * D * n_stochastic + MU * n_stochastic
    noise_amplitude = np.sqrt(np.maximum(0, noise_strength_sq) / (DX**2 * DT))
    
    eta_x = np.random.normal(0, 1, n_stochastic.shape)
    eta_y = np.random.normal(0, 1, n_stochastic.shape)
    
    flux_x = noise_amplitude * eta_x
    flux_y = noise_amplitude * eta_y
    
    grad_flux_y, _ = np.gradient(flux_y, DX, axis=(0, 1))
    _, grad_flux_x = np.gradient(flux_x, DX, axis=(0, 1))
    noise_term = grad_flux_x + grad_flux_y
    
    n_stochastic += (drift_term + noise_term) * DT

    if (step + 1) % save_interval == 0:
        history_det.append(n_deterministic.copy())
        history_sto.append(n_stochastic.copy())

print("Simulation finished. Creating animation...")

# --- 5. Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('black')
plt.style.use('dark_background')

def update(frame):
    for ax in axes:
        ax.clear()
            
    n_det = history_det[frame]
    n_sto = history_sto[frame]
    
    vmax = K * 1.1
    
    # --- Plot Deterministic Front ---
    axes[0].imshow(n_det, cmap='viridis', vmin=0, vmax=vmax, origin='lower', interpolation='bicubic')
    axes[0].set_title('Deterministic Front (DP/Mean-Field Theory)', color='white')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # --- Plot Stochastic Front ---
    axes[1].imshow(n_sto, cmap='viridis', vmin=0, vmax=vmax, origin='lower', interpolation='bicubic')
    axes[1].set_title('Stochastic Front (KMPI Theory)', color='white')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    
    fig.suptitle(f'Reaction-Diffusion Front Propagation (Time: {frame*DT*save_interval:.1f})', color='white', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return axes[0], axes[1]

ani = animation.FuncAnimation(fig, update, frames=len(history_det), interval=50, blit=False)
try:
    ani.save('reaction_front_comparison.gif', writer='pillow', fps=20)
    print("Animation saved as 'reaction_front_comparison.gif'.")
except Exception as e:
    print(f"Could not save animation: {e}")

plt.show()
