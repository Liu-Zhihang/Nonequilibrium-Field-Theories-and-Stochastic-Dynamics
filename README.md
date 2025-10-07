# Nonequilibrium Field Theories and Stochastic Dynamics

This repository contains learning notes and self-written code based on the YouTube lecture series "Nonequilibrium Field Theories and Stochastic Dynamics" taught by Prof. Erwin Frey at Ludwig-Maximilians-Universität München (LMU Munich) in Summer Semester 2025.

**Course Playlist:** [YouTube Playlist](https://www.youtube.com/watch?v=-pEPKnuN1iY&list=PL2IEUF-u3gRdSbgtuqH5RNTuT798s0GqX)

## Course Video Titles

Based on the course structure and repository contents, the lecture series includes the following topics:

### I. Foundations of Stochastic Processes (Lectures 1-10)

1. **Thermodynamics, Statistical Mechanics, Nonequilibrium Physics and Teaching Philosophy** (43 min)
2. **Simple Random Walk** (41 min)
3. **Gaussian Random Walk, Poisson Process, Gillespie Algorithm** (45 min)
4. **Gillespie Algorithm, Master Equation, Generating Function, Population Dynamics** (45 min)
5. **Population Dynamics: Linear Death Process, The Lotka-Volterra Process** (58 min)
6. **Fundamental Equations For Markov Processes: Chapman-Kolmogorov Equation** (36 min)
7. **Forward Master Eq., Q-matrix for Linear Birth-death Process, General Properties of Master Eq.** (41 min)
8. **Perron-Frobenius thm., Indecomposable Q-matrices, Rev. & Stationary Procs., Detailed Balance** (48 min)
9. **Consequences of Detailed Balance, Irreversibility and Entropy Production** (42 min)
10. **The Ehrenfest Model, Entropy and Kullback-Leibler Divergence** (37 min)

### II. Stochastic Dynamics of Particles (Lectures 11-25)

11. **Markov Chain Monte Carlo, Jump Processes, Diffusion Processes, Fokker-Planck Equation** (54 min)
12. **Brownian Motion (Wiener Process), Ornstein-Uhlenbeck Process, Einstein-Stokes Relation** (35 min)
13. **Monte Carlo Sampling as Stochastic Process** (44 min)
14. **Hamiltonian Monte Carlo Sampling** (49 min)
15. **Chemotaxis, Run-and-Tumble Motion as Two-State Process, Keller–Segel model** (47 min)
16. **Schnitzer Model, Anti-Diffusion, Motility-Induced Phase Separation** (33 min)
17. **Langevin Equation, Brownian Particle, Fluctuation–Dissipation Theorem** (43 min)
18. **Fokker-Planck Equation of Brownian Particle, Overdamped Langevin Equation, Smoluchowsky equation** (39 min)
19. **Path Integral Formulation of Langevin Equations** (45 min)
20. **Stochastic Differential Equations, Ito's Lemma, Stochastic integrals, Ito and Stratonovich** (42 min)
21. **Ito's Formulas, Transformation Between Stratonovich and Ito Formulation** (47 min)
22. **Path Integrals for Systems with Multiplicative Noise** (27 min)
23. **Interacting Brownian Particles, Fluctuations Near Equilibrium, Time Correlations of Fluctuations** (47 min)
24. **Onsager Coefficients and Symmetry Relations, Dynamic Form of the Fluctuation–Dissipation Theorem** (45 min)
25. **Gradient Dynamics, Model A, Classical Ising model, Ginzburg–Landau Equation, Allen–Cahn Equation** (54 min)

### III. From Discrete States to Fields (Lectures 26-35)

26. **Critical Slowing Down, Response Function, Dynamic Susceptibility, Model B** (39 min)
27. **Hydrodynamics of Simple Fluids, Frictionless Fluids, Euler Equation** (51 min)
28. **Viscous Fluid, Navier-Stokes Equation, Entropy Balance and Heat Conduction** (41 min)
29. **Irreversible Linear Thermodynamics, Dry Diffusive Particles Systems** (46 min)
30. **Brownian Particles Suspended in a Fluid, Model H** (30 min)
31. **Dynamic Functionals for Field Theories with Additive Noise, Onsager-Machlup Functional** (1h 3min)
32. **Janssen-De Dominicis Response Functional, Fluctuation-Dissipation Relation** (25 min)
33. **Non-Equilibrium Work and Fluctuation Theorem, Jarzynski's Work Relation, Crooks' Fluct. Theorem** (1h 19min)
34. **Directed Percolation, Spectral Method for Linear Death Process** (1h 6min)
35. **Path Integral for Master Equation** (31 min)

### IV. Field Theories of Nonequilibrium Systems (Lectures 36-40)

36. **Coherent State Path Integral, Operator Algebra and the Imaginary Noise** (40 min)
37. **Kramers-Moyal Expansion and the Low Noise Limit of the Path Integral** (41 min)
38. **Multi-Species Path Integrals, Rock-Paper-Scissors** (49 min)
39. **Path Integrals on a Lattice: From Hopping to Continuous Field Theories** (43 min)
40. **Kramers-Moyal Path Integral Approach, Field Theory for Interacting Particles** (32 min)

## Course Outline

![Course Outline](images/Course_Outline.jpg)

## Course Contents

This lecture series explores the fundamental principles and advanced concepts of nonequilibrium field theories and stochastic dynamics. The course focuses on stochastic processes in particle and field systems, emphasizing mathematical formalisms such as Langevin equations, Fokker-Planck equations, and path integrals. Additionally, the lectures cover applications in soft matter physics, active matter, and non-equilibrium statistical mechanics.

The course consists of 40 lectures organized according to the following curriculum structure:

### I. Foundations of Stochastic Processes (Lectures 1-10)

**1. Random Walks and Brownian Motion**
*Bernoulli/Gaussian walks, diffusion equation*

- **Lecture 1:** Thermodynamics, Statistical Mechanics, Nonequilibrium Physics and My Teaching Philosophy (43 min)
- **Lecture 2:** Simple Random Walk (41 min)
- **Lecture 3:** Gaussian Random Walk, Poisson Process, Gillespie algorithm (45 min)

**2. Elementary Stochastic Models**
*Poisson process, birth-death dynamics, molecular motors*

- **Lecture 4:** Gillespie Algorithm, Master Equation, Generating Function, Population Dynamics (45 min)
- **Lecture 5:** Population Dynamics: Linear Death Process, The Lotka-Volterra Process (58 min)

**3. Markov Processes and Master Equations**
*Chapman-Kolmogorov, discrete & continuous state spaces*

- **Lecture 6:** Fundamental Equations For Markov Processes: Chapman-Kolmogorov Equation (36 min)
- **Lecture 7:** Forward Master Eq., Q-matrix for Linear Birth-death Process, General Properties of Master Eq. (41 min)
- **Lecture 8:** Perron-Frobenius thm., Indecomposable Q-matrices, Rev. & Stationary Procs., Detailed Balance (48 min)
- **Lecture 9:** Consequences of Detailed Balance, Irreversibility and Entropy Production (42 min)
- **Lecture 10:** The Ehrenfest Model, Entropy and Kullback-Leibler Divergence (37 min)

### II. Stochastic Dynamics of Particles (Lectures 11-25)

**4. Langevin and Fokker-Planck Equations**
*Path integrals, multiplicative noise*

- **Lecture 11:** Markov Chain Monte Carlo, Jump Processes, Diffusion Processes, Fokker-Planck Equation (54 min)
- **Lecture 12:** Brownian Motion (Wiener Process), Ornstein-Uhlenbeck Process, Einstein-Stokes Relation (35 min)
- **Lecture 17:** Langevin Equation, Brownian Particle, Fluctuation–Dissipation Theorem (43 min)
- **Lecture 18:** Fokker-Planck Equation of Brownian Particle, Overdamped Langevin Equation, Smoluchowsky equation (39 min)
- **Lecture 19:** Path Integral Formulation of Langevin Equations (45 min)
- **Lecture 20:** Stochastic Differential Equations, Ito's Lemma, Stochastic integrals, Ito and Stratonovich (42 min)
- **Lecture 21:** Ito's Formulas, Transformation Between Stratonovich and Ito Formulation (47 min)
- **Lecture 22:** Path Integrals for Systems with Multiplicative Noise (27 min)

**5. Stochastic Simulation Techniques**
*Gillespie algorithm, stochastic integration*

- **Lecture 13:** Monte Carlo Sampling as Stochastic Process (44 min)
- **Lecture 14:** Hamiltonian Monte Carlo Sampling (49 min)
- **Lecture 15:** Chemotaxis, Run-and-Tumble Motion as Two-State Process, Keller–Segel model (47 min)
- **Lecture 16:** Schnitzer Model, Anti-Diffusion, Motility-Induced Phase Separation (33 min)

**6. Stochastic Thermodynamics (Trajectory Level)**
*Entropy production, detailed balance, fluctuation theorems*

- **Lecture 23:** Interacting Brownian Particles, Fluctuations Near Equilibrium, Time Correlations of Fluctuations (47 min)
- **Lecture 24:** Onsager Coefficients and Symmetry Relations, Dynamic Form of the Fluctuation–Dissipation Theorem (45 min)
- **Lecture 25:** Gradient Dynamics, Model A, Classical Ising model, Ginzburg–Landau Equation, Allen–Cahn Equation (54 min)

### III. From Discrete States to Fields (Lectures 26-35)

**7. Reaction Networks and Field Theories**
*Master equations, Kramers-Moyal expansion*

- **Lecture 34:** Directed Percolation, Spectral Method for Linear Death Process (1h 6min)
- **Lecture 35:** Path Integral for Master Equation (31 min)
- **Lecture 37:** Kramers-Moyal Expansion and the Low Noise Limit of the Path Integral (41 min)
- **Lecture 38:** Multi-Species Path Integrals, Rock-Paper-Scissors (49 min)
- **Lecture 39:** Path Integrals on a Lattice: From Hopping to Continuous Field Theories (43 min)

**8. Coarse-Grained Dynamics of Fields**
*Relaxational dynamics, conserved vs. non-conserved fields*

- **Lecture 26:** Critical Slowing Down, Response Function, Dynamic Susceptibility, Model B (39 min)
- **Lecture 27:** Hydrodynamics of Simple Fluids, Frictionless Fluids, Euler Equation (51 min)
- **Lecture 28:** Viscous Fluid, Navier-Stokes Equation, Entropy Balance and Heat Conduction (41 min)
- **Lecture 29:** Irreversible Linear Thermodynamics, Dry Diffusive Particles Systems (46 min)
- **Lecture 30:** Brownian Particles Suspended in a Fluid, Model H (30 min)

### IV. Field Theories of Nonequilibrium Systems (Lectures 31-40)

**9. Dynamical Functionals and MSR Formalism**
*Onsager-Machlup, Janssen-de Dominicis, path integrals*

- **Lecture 31:** Dynamic Functionals for Field Theories with Additive Noise, Onsager-Machlup Functional (1h 3min)
- **Lecture 32:** Janssen-De Dominicis Response Functional, Fluctuation-Dissipation Relation (25 min)
- **Lecture 36:** Coherent State Path Integral, Operator Algebra and the Imaginary Noise (40 min)

**10. Hydrodynamics and Fluctuating Fluids**
*Model H, momentum conservation, suspension dynamics*

- **Lecture 40:** Kramers-Moyal Path Integral Approach, Field Theory for Interacting Particles (32 min)

**11. Nonequilibrium Pattern Formation**
*Instabilities, non-equilibrium steady states*

- **Lecture 33:** Non-Equilibrium Work and Fluctuation Theorem, Jarzynski's Work Relation, Crooks' Fluct. Theorem (1h 19min)

**12. Field Theory of Active Matter**
*Self-propulsion, self-organisation*

*Note: Some topics from the theoretical outline may be distributed across multiple lectures or covered in integrated discussions within the video series.*

## Repository Structure

This repository contains Python implementations and Jupyter notebooks that illustrate the concepts covered in the course:

- **Random Walks and Stochastic Processes:** Simple random walk, Gaussian random walk, Poisson processes
- **Population Dynamics:** Birth-death processes, Lotka-Volterra models, Gillespie algorithm
- **Markov Processes:** Master equations, Q-matrices, detailed balance
- **Brownian Motion:** Wiener processes, Ornstein-Uhlenbeck processes, diffusion
- **Monte Carlo Methods:** MCMC, Hamiltonian Monte Carlo
- **Active Matter:** Run-and-tumble motion, chemotaxis, MIPS (Motility-Induced Phase Separation)
- **Field Theories:** Ising model, Ginzburg-Landau equations, critical dynamics

## Prerequisites

- Statistical mechanics and thermodynamics
- Probability theory and stochastic processes
- Differential equations
- Basic knowledge of field theory (helpful but not required)

## Usage

Each Python file corresponds to specific topics covered in the lecture series. The code serves as practical implementations of the theoretical concepts presented in the YouTube videos, developed as part of self-study and learning notes.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the [LICENSE](LICENSE) file for details.

**Summary:** You are free to share and adapt this material for non-commercial purposes, but you must give appropriate credit and indicate if changes were made.

---

**Note:** This repository is for educational purposes and contains study materials for the course "Nonequilibrium Field Theories and Stochastic Dynamics" by Prof. Erwin Frey at LMU Munich.




