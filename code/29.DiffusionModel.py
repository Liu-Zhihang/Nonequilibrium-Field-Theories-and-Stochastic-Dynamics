# --- 1. Preamble and Imports ---
import numpy as np
import torch # Using PyTorch for its GPU capabilities and autograd, convenient for ML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# --- 2. Setup and Hyperparameters ---
# Timesteps for the diffusion process
T = 200 
# Define the beta schedule (how much noise is added at each step)
# This is a linear schedule, which is simple and effective.
betas = torch.linspace(0.0001, 0.02, T)

# Pre-calculate alphas and their cumulative products, which are key to the diffusion math
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# These pre-calculated values are essential for both the forward and reverse processes

# --- 3. Create the "Ordered System": A Simple Image ---
def create_initial_state(size=64, shape='smiley'):
    """Creates a simple 2D image as our initial ordered state."""
    img = torch.zeros((size, size))
    if shape == 'smiley':
        # Face circle
        for i in range(size):
            for j in range(size):
                if (i - size//2)**2 + (j - size//2)**2 < (size//2.5)**2:
                    img[i, j] = 1.0
        # Eyes
        img[size//2-10:size//2-5, size//2-10:size//2-5] = -1.0
        img[size//2-10:size//2-5, size//2+5:size//2+10] = -1.0
        # Mouth
        for i in range(size//2+5, size//2+15):
            img[i, size//2-10:size//2+10] = -1.0
    return img

x_start = create_initial_state()

# --- 4. The Forward Process (Physics Perspective) ---
def q_sample(x_start, t, noise=None):
    """
    Adds noise to an image x_start to get its state at time t.
    This function directly implements the forward physical diffusion process.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Get the pre-calculated coefficients for the given timestep t
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
    
    # Apply the forward process formula
    # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

# --- 5. The Reverse Process (Generative AI Perspective) ---
def p_sample(model, x, t):
    """
    Performs one step of denoising using a 'model'.
    This is the core of the reverse generative process.
    """
    # Use the model to predict the noise that was added at this timestep
    predicted_noise = model(x, t)
    
    # The mathematical formula to reverse one step
    alpha_t = alphas[t]
    alpha_cumprod_t = alphas_cumprod[t]
    
    # Subtract the predicted noise from the current image
    denoised_x = (x - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
    
    # Add back a small amount of stochastic noise (optional but common in practice)
    if t > 0:
        noise = torch.randn_like(x)
        beta_t = betas[t]
        sigma_t = torch.sqrt(beta_t) # Simplified variance
        denoised_x += sigma_t * noise
        
    return denoised_x

# --- 6. The "Oracle" Model (Simulating a Perfectly Trained AI) ---
def oracle_model(x_t, t, x_start_ref):
    """
    This is a placeholder for a real, trained neural network (like a U-Net).
    An 'oracle' has access to the original image (x_start_ref) and can therefore
    perfectly calculate the noise that was added. This allows us to visualize
    the ideal reverse process without the need for model training.
    """
    # The true noise is calculated by rearranging the forward process formula
    true_noise = (x_t - torch.sqrt(alphas_cumprod[t]) * x_start_ref) / torch.sqrt(1. - alphas_cumprod[t])
    return true_noise

# --- 7. Run the Full Simulation and Store Data ---
print("Running forward and reverse simulations to generate data...")
# Timesteps to display in the GIF and plots
display_timesteps = [0, T//4, T//2, 3*T//4, T-1]

# --- Run Forward Process ---
forward_images = []
for t in range(T):
    forward_images.append(q_sample(x_start, t))

# --- Run Reverse Process ---
# We need a reference to the original image for our 'oracle_model'
oracle_with_ref = lambda x, t: oracle_model(x, t, x_start)

reverse_images = []
# Start the reverse process from pure noise
x_t = torch.randn_like(x_start)
reverse_images.append(x_t)

for t in reversed(range(T)):
    x_t = p_sample(oracle_with_ref, x_t, t)
    reverse_images.append(x_t)
reverse_images = list(reversed(reverse_images)) # Put them in chronological order
print("Simulations finished.")

# --- 8. Create and Save Visualization ---
print("Creating visualization GIF and plots...")
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 5)

# Create axes for the images
ax_f = [fig.add_subplot(gs[0, i]) for i in range(5)]
ax_r = [fig.add_subplot(gs[1, i]) for i in range(5)]

def update(frame):
    """Update function for the animation."""
    fig.clear()
    gs = fig.add_gridspec(2, 5)
    ax_f = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax_r = [fig.add_subplot(gs[1, i]) for i in range(5)]
    
    fig.suptitle('Physics Diffusion vs. Generative AI Diffusion', fontsize=16)

    # --- Update Forward Process Visualization ---
    for i, t_idx in enumerate(display_timesteps):
        # We show the gradual noising process
        current_t = min(t_idx, frame)
        ax_f[i].imshow(forward_images[current_t].numpy(), cmap='viridis', vmin=-2, vmax=2)
        ax_f[i].set_title(f'Forward t={current_t+1}')
        ax_f[i].axis('off')
    ax_f[0].set_ylabel('Forward Process\n(Order to Chaos)', fontsize=12, labelpad=20)

    # --- Update Reverse Process Visualization ---
    for i, t_idx in enumerate(display_timesteps):
        # We show the gradual denoising process
        current_t_rev = T - min(t_idx, frame)
        img_to_show = reverse_images[T - current_t_rev] if current_t_rev < len(reverse_images) else reverse_images[-1]
        ax_r[i].imshow(img_to_show.numpy(), cmap='viridis', vmin=-2, vmax=2)
        ax_r[i].set_title(f'Reverse t={T-current_t_rev+1}')
        ax_r[i].axis('off')
    ax_r[0].set_ylabel('Reverse Process\n(Chaos to Order)', fontsize=12, labelpad=20)
    
    return fig,

# Create and save the animation
anim = animation.FuncAnimation(fig, update, frames=T, interval=50, blit=False)
anim.save('physics_vs_ai_diffusion.gif', writer='pillow', fps=20)
plt.close(fig) # Close the animation figure

# --- Create Final Distribution Plot ---
fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
ax_dist.set_title('Evolution of Pixel Value Distribution')
ax_dist.set_xlabel('Pixel Value')
ax_dist.set_ylabel('Probability Density')

# Plot initial distribution
initial_pixels = x_start.flatten().numpy()
ax_dist.hist(initial_pixels, bins=50, range=(-2, 2), density=True, histtype='step', lw=2, label='Initial State (Ordered)')

# Plot noise distribution (final state of forward process)
noise_pixels = forward_images[-1].flatten().numpy()
ax_dist.hist(noise_pixels, bins=50, range=(-2, 2), density=True, histtype='step', lw=2, label='Fully Diffused State (Chaos)')

# Plot a standard normal distribution for reference
x_norm = np.linspace(-2, 2, 100)
y_norm = norm.pdf(x_norm, 0, 1)
ax_dist.plot(x_norm, y_norm, 'r--', label='Standard Normal (Gaussian)')

ax_dist.legend()
ax_dist.grid(True, alpha=0.3)
plt.show()
print("Visualization saved and plot displayed.")
