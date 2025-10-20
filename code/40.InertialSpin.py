import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree

def simulate_inertial_flocking(
    num_particles=400, box_size=100.0, v0=0.5,
    R=10.0, dt=0.2, total_time=400.0
):
    """
    Simulates the Inertial Spin Model for collective motion,
    showcasing realistic turning waves.
    """
    # Model parameters
    inertia = 0.5  # Moment of inertia
    friction = 0.2 # Rotational friction
    align_strength = 0.5 # How strongly individuals align

    # Initialization
    positions = np.random.rand(num_particles, 2) * box_size
    orientations = np.random.rand(num_particles) * 2 * np.pi
    angular_velocities = np.zeros(num_particles)

    # --- Animation Setup ---
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xticks([])
    ax.set_yticks([])

    quiver = ax.quiver(
        positions[:, 0], positions[:, 1],
        np.cos(orientations), np.sin(orientations),
        color='cyan', scale=40, headwidth=4, pivot='middle'
    )
    title = ax.set_title("Disordered State | t = 0.0", color='white', fontsize=16)

    noise_strength = 0.1 # Constant low noise

    def update(frame):
        nonlocal positions, orientations, angular_velocities

        # --- Find neighbors using KDTree ---
        tree = KDTree(positions, boxsize=[box_size, box_size])
        neighbor_indices = tree.query_ball_point(positions, r=R)

        # --- Calculate Alignment Torques ---
        align_torques = np.zeros(num_particles)
        for i in range(num_particles):
            neighbors = neighbor_indices[i]
            if len(neighbors) > 1:
                # Calculate the vector difference to the mean orientation
                # This is a more stable way to calculate the torque
                mean_vec = np.array([
                    np.mean(np.cos(orientations[neighbors])),
                    np.mean(np.sin(orientations[neighbors]))
                ])
                # The torque is proportional to the "cross product" in 2D
                # (sin(theta_mean - theta_i))
                current_vec = np.array([np.cos(orientations[i]), np.sin(orientations[i])])
                # Normalize mean_vec to avoid magnitude effects
                mean_vec /= np.linalg.norm(mean_vec)
                # Torque is sin of angle difference: u x v = u_x v_y - u_y v_x
                torque = current_vec[0]*mean_vec[1] - current_vec[1]*mean_vec[0]
                align_torques[i] = align_strength * torque

        # --- Update Angular Velocities (Second-order dynamics) ---
        random_torques = noise_strength * (np.random.rand(num_particles) - 0.5)
        angular_accelerations = (
            -friction * angular_velocities + align_torques + random_torques
        ) / inertia
        angular_velocities += angular_accelerations * dt

        # --- Update Orientations and Positions ---
        orientations += angular_velocities * dt
        positions[:, 0] += v0 * np.cos(orientations) * dt
        positions[:, 1] += v0 * np.sin(orientations) * dt
        
        # Periodic boundary conditions
        positions %= box_size

        # --- Update Visualization ---
        quiver.set_offsets(positions)
        quiver.set_UVC(np.cos(orientations), np.sin(orientations))
        
        title.set_text(f"Inertial Flocking | t = {frame*dt:.1f}")
        return quiver, title

    ani = FuncAnimation(
        fig, update, frames=int(total_time / dt),
        blit=True, interval=20
    )
    ani.save('inertial_flocking.gif', writer='pillow', fps=30)
    plt.show()

# Run the simulation
simulate_inertial_flocking(num_particles=300, R=8.0)