import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from hypernet_viscous import newton_raphson, viscous_burgers_res, viscous_burgers_jac

@jax.jit
def gudonov_flux(u_L, u, u_R):
    return 0.5*jnp.where(u > 0, u**2 - u_L**2, u_R**2 - u**2)

@jax.jit
def viscous_burgers_res_jax(w, u, grid, dt, wp, mu, B):
    """
    Returns a residual vector for the 1d inviscid burgers equation using a first-order
    Godunov space discretization and a 2nd-order trapezoid rule time integrator
    """
    dx = grid[1] - grid[0]

    # f = jnp.zeros(grid.size)
    # fp = jnp.zeros(grid.size)
    # f = f.at[1:].set(0.5 * jnp.square(w))
    # f = f.at[0].set(f[-1])
    # fp = fp.at[1:].set(0.5 * jnp.square(wp))
    # fp = fp.at[0].set(fp[-1])
    # # f_G = gudonov_flux(jnp.roll(w, 1), w)
    # # fp_G = gudonov_flux(jnp.roll(wp, 1), wp)
    # # print('rom  |val|: {}'.format(jnp.linalg.norm(f[1:] - f[:-1]) * 100))
    # r = w - wp + 0.5*(dt/dx)*((fp[1:]-fp[:-1]) + (f[1:] - f[:-1])) - \
    #     0.5*mu[0]*(dt/dx/dx)*(jnp.roll(wp, 1) - 2*wp + jnp.roll(wp, -1) + jnp.roll(w, 1) - 2*w + jnp.roll(w, -1)) \
    #     - dt*jnp.dot(B, u)

    f_G = gudonov_flux(jnp.roll(w, 1), w, jnp.roll(w, -1))
    fp_G = gudonov_flux(jnp.roll(wp, 1), wp, jnp.roll(wp, -1))
    # print('rom  |val|: {}'.format(jnp.linalg.norm(f[1:] - f[:-1]) * 100))
    r = w - wp + 0.5 * (dt / dx) * (fp_G + f_G) - \
        0.5 * mu[0] * (dt / dx / dx) * (
                jnp.roll(wp, 1) - 2 * wp + jnp.roll(wp, -1) + jnp.roll(w, 1) - 2 * w + jnp.roll(w, -1)) \
        - dt * jnp.dot(B, u)
    return r

@jax.jit
def viscous_burgers_res_jax_hrom(q, u, grid, dt, qp, mu, B, V, idxs):
    """
    Returns a residual vector for the 1d inviscid burgers equation using a first-order
    Godunov space discretization and a 2nd-order trapezoid rule time integrator
    """
    dx = grid[1] - grid[0]

    w = V @ q
    wp = V @ qp

    mask = idxs
    mask_l = jnp.mod(idxs - 1, V.shape[0])
    mask_r = jnp.mod(idxs + 1, V.shape[0])

    w_m = w[mask]
    w_l = w[mask_l]
    w_r = w[mask_r]
    wp_m = wp[mask]
    wp_l = wp[mask_l]
    wp_r = wp[mask_r]

    f_G = gudonov_flux(w_l, w_m, w_r)
    fp_G = gudonov_flux(wp_l, wp_m, wp_r)

    # f = jnp.zeros(w.shape[0])
    # fp = jnp.zeros_like(f)
    # f = f.at[0:].set(0.5 * jnp.square(w))
    # fp = fp.at[0:].set(0.5 * jnp.square(wp))

    # r = w[mask] - wp[mask] + 0.5*(dt/dx)*((fp[mask]-fp[mask_l]) + (f[mask] - f[mask_l])) - \
    #     0.5*mu[0]*(dt/dx/dx)*(wp[mask_l] - 2*wp[mask] + wp[mask_r] + w[mask_l] - 2*w[mask] + w[mask_r]) \
    #     - dt * jnp.dot(B[mask, :], u)
    r = w[mask] - wp[mask] + 0.5 * (dt / dx) * (fp_G + f_G) - \
        0.5 * mu[0] * (dt / dx / dx) * (wp[mask_l] - 2 * wp[mask] + wp[mask_r] + w[mask_l] - 2 * w[mask] + w[mask_r]) \
        - dt * jnp.dot(B[mask, :], u)
    return r

# @jax.jit
def viscous_burgers_implicit_step(grid, w0, u, dt, mu, B):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a parameterized inviscid 1D burgers problem.
    The parameters are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """
    @jax.jit
    def res(w, u):
        return viscous_burgers_res_jax(w, u, grid, dt, w0, mu, B)

    @jax.jit
    def jac(w, u):
        return jax.jacobian(viscous_burgers_res_jax, 0)(w, u, grid, dt, w0, mu, B)

    w, _ = newton_raphson_jax(res, jac, w0, u, max_its=50)

    return w

def newton_raphson_jax(func, jac, x0, u, max_its=20, relnorm_cutoff=1e-12):
    x = x0.copy()
    init_norm = jnp.linalg.norm(func(x0, u))
    resnorms = []
    for i in range(max_its):
        resnorm = jnp.linalg.norm(func(x, u))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        J = jac(x, u)
        f = func(x, u)
        x -= jnp.linalg.solve(J, f)
    return x, resnorms

#@jax.jit
def viscous_burgers_implicit_step_rom(grid, w0, u, dt, mu, V):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a parameterized inviscid 1D burgers problem.
    The parameters are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """

    def res(w, u):
        return viscous_burgers_res_jax(w, u, grid, dt, w0, mu, B)

    def jac(w, u):
        return jax.jacobian(viscous_burgers_res_jax, 0)(w, u, grid, dt, w0, mu, B)

    w, _ = newton_raphson_jax_rom(res, jac, w0, u, V, max_its=50)

    return w

def newton_raphson_jax_rom(func, jac, x0, u, V, max_its=20, relnorm_cutoff=1e-5, min_delta=1e-1):
    q = V.T @ x0
    init_norm = jnp.linalg.norm(func(x0, u))
    resnorms = []
    for i in range(max_its):
        x = V @ q
        resnorm = jnp.linalg.norm(func(x, u))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        J = jac(x, u) @ V
        f = func(x, u)
        q -= jnp.linalg.lstsq(J, f)[0]
    return x, resnorms


# @jax.jit
def viscous_burgers_implicit_step_hrom(grid, q0, u, dt, mu, B, V, weights):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a parameterized inviscid 1D burgers problem.
    The parameters are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """

    mask = jnp.array(weights) > 0
    idxs = jnp.arange(weights.shape[0])[mask]

    def res(q, u):
        # val1 = viscous_burgers_res_jax(V@q, u, grid, dt, V@q0, mu)
        # val2 = viscous_burgers_res_jax_hrom(q, u, grid, dt, q0, mu, V, weights)
        # print('diff: {}'.format(jnp.linalg.norm(val1 - val2)))
        return viscous_burgers_res_jax_hrom(q, u, grid, dt, q0, mu, B, V, idxs)

    def jac(q, u):
        return jax.jacobian(viscous_burgers_res_jax_hrom, 0)(q, u, grid, dt, q0, mu, B, V, idxs)

    q, _ = newton_raphson_jax_hrom(res, jac, q0, u, V, weights, max_its=10)

    return q

def newton_raphson_jax_hrom(func, jac, x0, u, V, weights, max_its=20, relnorm_cutoff=1e-5,
                            min_delta=1e-1):
    q = x0
    mask = weights > 0
    w = weights[mask]
    init_norm = jnp.linalg.norm(w * func(x0, u))
    resnorms = []
    for i in range(max_its):
        x = q
        resnorm = jnp.linalg.norm(w * func(x, u))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        J = jnp.diag(w) @ jac(x, u)# @ V[mask, :]
        # J = (w * jac(x, u).T).T
        f = func(x, u) * w
        q -= jnp.linalg.lstsq(J, f)[0]
    # print('i = {}, relative norm: {}'.format(i, resnorm/init_norm))
    return q, resnorms


def viscous_burgers_implicit_jit(grid, w0, dt, num_steps, mu):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve a parameterized inviscid 1D burgers problem.
    The parameters are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """

    snaps = jnp.zeros((w0.size, num_steps+1))
    snaps[:, 0] = w0
    wp = w0.copy()
    for i in range(num_steps):

        def res(w):
            return viscous_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            return viscous_burgers_jac(w, grid, dt, mu)

        print(" ... Working on timestep {}".format(i))
        w, resnorms = newton_raphson(res, jac, wp, max_its=50)

        # if i %  == 0:
        #     plt.plot((grid[1:] + grid[:-1])/2, w)
        #     plt.show()
        #     time.sleep(0.1)

        snaps[:, i+1] = w.copy()
        wp = w.copy()

    return snaps

# @jax.jit
def compute_ECSW_training_matrix(snaps, prev_snaps, basis, grid, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """

    n_x = snaps.shape[0]
    n_u = 5

    # Construct linear control matrix
    interval = int(n_x / n_u)
    eye = jnp.eye(n_u, n_u)
    B = jnp.kron(eye, jnp.ones(interval,)).T
    u = np.zeros((n_u, ))



    n_hdm, n_snaps = snaps.shape
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))
    projector = basis.dot(basis.T)
    for isnap in range(n_snaps):
        snap = prev_snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        u_proj = projector.dot(snap)
        up_proj = projector.dot(uprev)
        ires = viscous_burgers_res_jax(u_proj, u, grid, dt, up_proj, mu, B) #res(snap, grid, dt, uprev, mu)
        Ji = jax.jacobian(viscous_burgers_res_jax, 0)(u_proj, u, grid, dt, up_proj, mu, B) #jac(snap, grid, dt)
        Wi = Ji.dot(basis)
        for inode in range(n_hdm):
            C[isnap*n_pod:isnap*n_pod+n_pod, inode] = ires[inode]*Wi[inode]

    return C
