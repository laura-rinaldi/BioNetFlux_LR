import sys
import os
import numpy as np
from typing import Callable, List, Optional, Union


# Add the path to folder B
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from bionetflux.core.problem import Problem

# Create Keller-Segel chemotaxis problem
ks_problem = Problem(
    neq=2,
    domain_start=0.0,
    domain_length=1.0,
    parameters=np.array([2.0, 1.0, 0.1, 1.5]),  # [mu, nu, a, b]
    problem_type="keller_segel",
    name="chemotaxis_problem"
)

# Set chemotactic sensitivity
ks_problem.set_chemotaxis(
    chi=lambda phi: 1.0 / (1.0 + phi**2),
    dchi=lambda phi: -2.0 * phi / (1.0 + phi**2)**2
)

# Set initial conditions
ks_problem.set_initial_condition(0, lambda s: np.exp(-(s-0.5)**2/0.1))  # u
ks_problem.set_initial_condition(1, lambda s: np.ones_like(s))          # phi

# Set source terms
ks_problem.set_force(0, lambda s, t: 0.1 * np.exp(-t) * np.sin(np.pi*s))
ks_problem.set_force(1, lambda s, t: np.zeros_like(s))

# Set boundary conditions (zero flux)
for eq_idx in range(2):
    ks_problem.set_boundary_flux(eq_idx, 
                                left_flux=lambda t: 0.0,
                                right_flux=lambda t: 0.0)