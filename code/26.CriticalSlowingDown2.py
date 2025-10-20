import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Parameters ---
L = 1.0
c = 1.0
q_fixed = 0.1  # We fix the wavevector q as shown on the blackboard

# --- Define three scenarios to show the transition ---
r_far = 1.0       # Far from critical point (gentle curve)
r_inter = 0.2     # Intermediate case
r_near = 0.01     # Very near critical point (sharp peak)

# --- Calculate the corresponding relaxation times ---
tau_q_far = 1.0 / (L * (r_far + c * q_fixed**2))
tau_q_inter = 1.0 / (L * (r_inter + c * q_fixed**2))
tau_q_near = 1.0 / (L * (r_near + c * q_fixed**2))

# Setup the frequency omega range
omega = np.linspace(-4, 4, 1000)

# --- 2. Define functions for chi' and chi'' ---
def chi_prime(omega, L, tau_q):
    """Calculates the real part of the dynamic susceptibility."""
    return (L * tau_q) / (1 + (omega * tau_q)**2)

def chi_double_prime(omega, L, tau_q):
    """Calculates the imaginary part of the dynamic susceptibility."""
    return (L * omega * tau_q**2) / (1 + (omega * tau_q)**2)

# --- 3. Calculate data for all scenarios ---
chi_p_far = chi_prime(omega, L, tau_q_far)
chi_pp_far = chi_double_prime(omega, L, tau_q_far)

chi_p_inter = chi_prime(omega, L, tau_q_inter)
chi_pp_inter = chi_double_prime(omega, L, tau_q_inter)

chi_p_near = chi_prime(omega, L, tau_q_near)
chi_pp_near = chi_double_prime(omega, L, tau_q_near)

# --- 4. Visualization (replicating the blackboard concept) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Dynamic Susceptibility Spectrum (Fixed q)', fontsize=18)

# --- Plot for chi' (Real Part - Storage Response) ---
ax1.plot(omega, chi_p_far, label=f'Far from criticality ($r={r_far}$)', lw=2)
ax1.plot(omega, chi_p_inter, label=f'Intermediate ($r={r_inter}$)', lw=2, linestyle='--')
ax1.plot(omega, chi_p_near, label=f'Near criticality ($r={r_near}$)', lw=2)
ax1.set_title(r"Real Part $\chi'(\omega)$ (Storage Response)", fontsize=14)
ax1.set_xlabel(r'Angular Frequency $\omega$', fontsize=12)
ax1.set_ylabel(r"Response Magnitude $\chi'$", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()
ax1.axhline(0, color='black', lw=0.5)
ax1.axvline(0, color='black', lw=0.5)

# --- Plot for chi'' (Imaginary Part - Dissipative Response) ---
ax2.plot(omega, chi_pp_far, label=f'Far from criticality, width $\sim 1/\\tau_q \\approx {1/tau_q_far:.2f}$', lw=2)
ax2.plot(omega, chi_pp_inter, label=f'Intermediate, width $\sim 1/\\tau_q \\approx {1/tau_q_inter:.2f}$', lw=2, linestyle='--')
ax2.plot(omega, chi_pp_near, label=f'Near criticality, width $\sim 1/\\tau_q \\approx {1/tau_q_near:.2f}$', lw=2)
ax2.set_title(r"Imaginary Part $\chi''(\omega)$ (Dissipative Response)", fontsize=14)
ax2.set_xlabel(r'Angular Frequency $\omega$', fontsize=12)
ax2.set_ylabel(r"Dissipation Magnitude $\chi''$", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()
ax2.axhline(0, color='black', lw=0.5)
ax2.axvline(0, color='black', lw=0.5)

# Set a symmetric y-limit for the chi'' plot to better see the change in width
max_y_val = np.max(chi_pp_inter) * 1.2
ax2.set_ylim(-max_y_val, max_y_val)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()