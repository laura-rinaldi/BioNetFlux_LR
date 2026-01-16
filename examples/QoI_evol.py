
def pde_solver_integral(x, sol):
    """
    Compute the QoI integral at a prescribed time using the composite
    trapezoidal rule.

    Parameters
    ----------
    x : array-like
        Mesh points in space (row or column vector).
    sol : array-like
        Solution values at the mesh points.

    Returns
    -------
    I : float
        Value of the integral.
    """

    # Compute mesh size (vector of spacings)
    h =  np.diff(x)[0]
    # Composite trapezoidal rule:
    # sum over h[i] * (sol[i] + sol[i+1]) / 2
    
    I = np.sum(h * (sol[:-1] + sol[1:]) / 2)

    return I



def pde_solver_barycenter(x, sol):
    """
    Compute the barycenter (center of mass) of a PDE solution.

    Parameters
    ----------
    x : 1D array
        Spatial mesh (row or column vector).
    sol : 1D array
        Solution values at the mesh points.

    Returns
    -------
    B : float
        Barycenter of the solution.
    """

    # Mesh size
    h =  np.diff(x)[0]
    x_tile = np.tile(x, 4)

    # Integral of the solution (composite trapezoidal rule)
    I = np.sum(h * (sol[0:] + sol[0:]) / 2)

    # Barycenter numerator (Simpson-like rule)
    fa = x_tile[:-1] * sol[:-1]
    fb = x_tile[1:]  * sol[1:]
    fc = (x_tile[:-1] + x_tile[1:]) * (sol[:-1] + sol[1:]) / 4

    numerator = np.sum(h * (fa + fb + 4 * fc) / 6)

    return numerator / I

