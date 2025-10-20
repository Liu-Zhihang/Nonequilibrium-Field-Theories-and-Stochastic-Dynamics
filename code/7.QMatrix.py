import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_birth_death_q_matrix(N, mu, lam):
    """
    Construct the Q matrix for a linear birth-death process.

    Parameters:
    N (int): Size of the state space (from 0 to N-1). This is a truncation, as theoretically the state space is infinite.
    mu (float): Birth rate per individual.
    lam (float): Death rate per individual (lambda is a keyword in python, so we use lam).

    Returns:
    numpy.ndarray: Q matrix of size (N, N).
    """
    # Initialize an N x N zero matrix
    Q = np.zeros((N, N))

    # Fill non-diagonal elements
    for n in range(1, N):
        # Death process: n -> n-1
        # Q_{n-1, n} = w(n-1, n) = lambda * n
        Q[n-1, n] = lam * n
        # Birth process: n-1 -> n
        # Q_{n, n-1} = w(n, n-1) = mu * (n-1)
        Q[n, n-1] = mu * (n-1)

    # Fill diagonal elements
    # Q_{n,n} = - (exit rate) = - (birth rate + death rate)
    for n in range(N):
        # Exit rate is the negative sum of all non-diagonal elements in this column
        # This is also equivalent to - (mu * n + lam * n)
        # Note: The row sum of Q matrix is 0, but here we follow the professor's board notation,
        # where transition rates are placed in Q_{destination, source}, so the column sum is 0.
        # If it's the standard mathematical notation Q_{i,j} representing transition from j->i, then the column sum is 0.
        # If it's the common notation in probability theory Q_{i,j} representing transition from i->j, then the row sum is 0.
        # The implementation here follows Q[row, col] = Q_{row, col}, i.e., transition from col to row, so the column sum is 0.
        # Q_{n,n} = - sum_{m!=n} Q_{m,n}
        Q[n, n] = -np.sum(Q[:, n])

    return Q

# --- Parameter settings ---
N_states = 15  # State space size (0, 1,..., 14)
mu = 1.0       # Birth rate
lambda_rate = 1.2 # Death rate

# Create Q matrix
Q_matrix = create_birth_death_q_matrix(N_states, mu, lambda_rate)

# --- Visualization ---
plt.figure(figsize=(10, 8))
sns.heatmap(Q_matrix, annot=True, fmt=".1f", cmap="vlag", linewidths=.5, cbar=True)
plt.title(f'Q-Matrix for Linear Birth-Death Process (N={N_states}, μ={mu}, λ={lambda_rate})', fontsize=16)
plt.xlabel('Source State (m)', fontsize=12)
plt.ylabel('Destination State (n)', fontsize=12)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.show()

print("Q Matrix (truncated):")
print(np.round(Q_matrix, 2))