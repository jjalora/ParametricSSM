"""
Use the Burgers equation to try out some learning-based hyper-reduction approaches
"""

import glob
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sklearn.cluster as clust
import jax
import jax.numpy as jnp

import pdb

def make_1D_grid(x_low, x_up, num_cells):
    """
    Returns a 1d ndarray of cell boundary points between a lower bound and an upper bound
    with the given number of cells
    """
    grid = np.linspace(x_low, x_up, num_cells+1)
    return grid

def viscous_burgers_explicit(grid, w0, dt, num_steps, mu):
    """
    Use a first-order Godunov spatial discretization and a first-order forward Euler time
    integrator to solve a parameterized viscous 1D burgers problem.
    The parameters
    are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """

    snaps = np.zeros((w0.size, num_steps+1))
    snaps[:, 0] = w0
    wp = w0.copy()
    dx = grid[1] - grid[0]
    xc = (grid[1:] + grid[:-1])/2
    f = np.zeros(grid.size)
    for i in range(num_steps):
        f[1:] = 0.5 * np.square(wp)
        f[0] = f[-1]
        w = wp - dt * (f[1:] - f[:-1]) / dx + mu[0]*dt*(np.roll(wp, 1) - 2*wp + np.roll(wp, -1))/dx/dx
        # if i % 50 == 0:
        #     plt.plot(xc, wp)
        #     plt.show()
        #     time.sleep(1)
        snaps[:, i + 1] = w
        wp = w.copy()
    return snaps

def viscous_burgers_implicit(grid, w0, dt, num_steps, mu):
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

    print("Running HDM for mu1={}".format(mu[0]))
    snaps = np.zeros((w0.size, num_steps+1))
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

def viscous_burgers_LSPG(grid, w0, dt, num_steps, mu, basis):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG PROM for a parameterized viscous 1D burgers problem
    with a source term. The parameters are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    npod = basis.shape[1]
    snaps =  np.zeros((w0.size, num_steps+1))
    red_coords = np.zeros((npod, num_steps+1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:,0] = w0
    red_coords[:,0] = y0
    wp = w0.copy()
    yp = y0.copy()
    print("Running ROM of size {} for mu1={}, mu2={}".format(npod, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w): 
            return viscous_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            return viscous_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_LSPG(res, jac, basis, yp)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep
        
        w = basis.dot(y)

        red_coords[:,i+1] = y.copy()
        snaps[:,i+1] = w.copy()
        wp = w.copy()
        yp = y.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)

def viscous_burgers_ecsw(grid, weights, w0, dt, num_steps, mu, basis):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an ECSW HPROM for a parameterized viscous 1D burgers problem
    with a source term. The parameters are as follows:
    mu[0]: diffusion coefficient

    so the equation solved is
    w_t + (0.5 * w^2)_x - mu[0]*w_xx= 0
    w(x=grid[0], t) = 0
    w(x, t=0) = w0
    """

    npod = basis.shape[1]
    snaps =  np.zeros((w0.size, num_steps+1))
    red_coords = np.zeros((npod, num_steps+1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:,0] = w0
    red_coords[:,0] = y0
    wp = w0.copy()
    wtmp = np.zeros_like(w0)
    yp = y0.copy()
    sample_inds, = np.where(weights != 0)
    sample_weights = weights[sample_inds]
    nsamp = sample_weights.size

    print("Running HROM of size {} with {} sample nodes for mu1={}, mu2={}".format(npod, nsamp, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w): 
            # return inviscid_burgers_ecsw_res(w, grid, sample_inds, dt, wp, mu)
            return viscous_burgers_res(w, grid, dt, wp, mu)

        def jac(w):
            # return inviscid_burgers_ecsw_jac(w, grid, sample_inds, dt)
            return viscous_burgers_jac(w, grid, dt)

        print(" ... Working on timestep {}".format(i))
        y, resnorms = gauss_newton_ECSW(res, jac, basis, yp, wtmp, sample_inds, sample_weights)
        w = basis.dot(y)

        red_coords[:,i+1] = y.copy()
        snaps[:,i+1] = w.copy()
        wp = w.copy()
        yp = y.copy()

    return snaps

def viscous_burgers_ecsw_res(w, grid, sample_inds, dt, wp, mu):
    """ 
    Returns a residual vector for the ECSW hyper-reduced 1d inviscid burgers equation
    using a first-order Godunov space discretization and a 2nd-order trapezoid rule time
    integrator
    Note: left and right boundary must be included
    """
    dx = grid[sample_inds + 1] - grid[sample_inds]

    fl = 0.5 * np.square(w[np.mod(sample_inds - 1, len(w))])
    flp = 0.5 * np.square(wp[np.mod(sample_inds - 1, len(w))])
    fr = 0.5 * np.square(w[sample_inds])
    frp = 0.5 * np.square(wp[sample_inds])
    wl = w[np.mod(sample_inds - 1, len(w))]
    wr = w[np.mod(sample_inds + 1, len(w))]
    wpl = wp[np.mod(sample_inds - 1, len(wp))]
    wpr = wp[np.mod(sample_inds + 1, len(wp))]
    r = w[sample_inds] - wp[sample_inds] + 0.5 * (dt / dx) * ((frp - flp) + (fr - fl)) - \
        0.5 * mu[0] * (dt / dx / dx) * ( wl - 2*w + wr + wpl - 2*wp + wpr)
    return r

def viscous_burgers_ecsw_jac(w, grid, sample_inds, dt, mu):
    """ 
    Returns a Jacobian for the ECSW hyper-reduced 1d inviscid burgers equation
    using a first-order Godunov space discretization and a 2nd-order trapezoid rule time
    integrator
    Note: left and right boundary must be included
    """
    n_samp = sample_inds.size
    dx = grid[sample_inds+1] - grid[sample_inds]
    xc = (grid[sample_inds+1] + grid[sample_inds])/2

    J = sp.lil_matrix((n_samp, n_samp))
    J += sp.eye(n_samp)
    J += 0.5*sp.diags( (dt/dx)*w[sample_inds])
    J[0, -1] += (dt/dx[0]*w[-1])
    J += sp.diags(2*mu[0]*dt/np.square(dx))
    J -= sp.diags(mu[0]*dt/np.square(dx[:-1]), -1)
    J -= sp.diags(mu[0]*dt/np.square(dx[1:]),   1)
    J[0, -1] -= mu[0]*dt/dx[-1]/dx[-1]
    J[-1, 0] -= mu[0]*dt/dx[0] /dx[0]
    for i, ind in enumerate(sample_inds):
        if ind+1 in sample_inds:
            J[i+1, i] = -0.5*w[ind]*dt/dx[i+1]
    return J.tocsr()

def viscous_burgers_res(w, grid, dt, wp, mu):
    """ 
    Returns a residual vector for the 1d inviscid burgers equation using a first-order
    Godunov space discretization and a 2nd-order trapezoid rule time integrator
    """
    dx = grid[1] - grid[0]

    f = np.zeros(grid.size)
    fp = np.zeros(grid.size)
    f[1:] = 0.5 * np.square(w)
    f[0] = f[-1]
    fp[1:] = 0.5 * np.square(wp)
    fp[0] = fp[-1]
    r = w - wp + 0.5*(dt/dx)*((fp[1:]-fp[:-1]) + (f[1:] - f[:-1])) - \
        0.5*mu[0]*(dt/dx/dx)*(np.roll(wp, 1) - 2*wp + np.roll(wp, -1) + np.roll(w, 1) - 2*w + np.roll(w, -1))
    return r

def viscous_burgers_jac(w, grid, dt, mu):
    """ 
    Returns a sparse Jacobian for the 1d inviscid burgers equation using a first-order
    Godunov space discretization and a 2nd-order trapezoid rule time integrator
    """
    dx = grid[1:] - grid[:-1]
    xc = (grid[1:] + grid[:-1])/2

    J = sp.lil_matrix((xc.size, xc.size))
    J += sp.eye(xc.size)
    J += 0.5*sp.diags((dt/dx)*w)
    J -= 0.5*sp.diags((dt/dx[1:])*w[:-1], -1)
    J[0, -1] += (dt/dx[0]*w[-1])
    J += sp.diags(2*mu[0]*dt/np.square(dx))
    J -= sp.diags(mu[0]*dt/np.square(dx[:-1]), -1)
    J -= sp.diags(mu[0]*dt/np.square(dx[1:]),   1)
    J[0, -1] -= mu[0]*dt/dx[-1]/dx[-1]
    J[-1, 0] -= mu[0]*dt/dx[0] /dx[0]
    return J.tocsr()

def newton_raphson(func, jac, x0, max_its=20, relnorm_cutoff=1e-12):
    x = x0.copy()
    init_norm = np.linalg.norm(func(x0))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(x))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        J = jac(x)
        f = func(x)
        x -= sp.linalg.spsolve(J, f)
    return x, resnorms

def gauss_newton_LSPG(func, jac, basis, y0, 
                      max_its=20, relnorm_cutoff=1e-5, min_delta=0.1):
    jac_time = 0
    res_time = 0
    ls_time = 0
    y = y0.copy()
    w = basis.dot(y0)
    init_norm = np.linalg.norm(func(w))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0
        t0 = time.time()
        f = func(w)
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(basis)
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)
        ls_time += time.time() - t0
        y += dy
        w = basis.dot(y)
        
    return y, resnorms, (jac_time, res_time, ls_time)

def line_search(func, y0, dy, step_size, 
                min_step=1e-10, shrink_factor=2, expand_factor=1.5):
  f0 = func(y0)
  step_size *= expand_factor
  while step_size >= min_step:
    f = func(y0 + step_size*dy)
    if f < f0:
      return y0 + step_size*dy, step_size
    else:
      step_size /= shrink_factor

  return y0 + step_size*dy, step_size


def gauss_newton_ECSW(func, jac, basis, y0, w, sample_inds, sample_weights,
                      stepsize=1, max_its=20, relnorm_cutoff=1e-4, min_delta=1E-8):
    y = y0.copy()
    w = basis.dot(y0)
    init_norm = np.linalg.norm(func(w)[sample_inds] * sample_weights)
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w)[sample_inds] * sample_weights)
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        J = jac(w).toarray()
        JV = J.dot(basis)[sample_inds, :]
        JVw = np.diag(sample_weights).dot(JV)

        f = func(w)[sample_inds]
        fw = f * sample_weights
        dy = np.linalg.lstsq(JVw, -fw, rcond=None)[0]
        # redjac = JV.T.dot(JV)
        # fred = JV.T.dot(f)
        # dy = np.linalg.solve(redjac, -fred)
        y += stepsize*dy
        w = basis.dot(y)

    return y, resnorms

def POD(snaps):
    u, s, vh = np.linalg.svd(snaps, full_matrices=False)
    return u, s

def podsize(svals, energy_thresh=None, min_size=None, max_size=None):
    """ Returns the number of vectors in a basis that meets the given criteria """

    if (energy_thresh is None) and (min_size is None) and (max_size is None):
        raise RuntimeError('Must specify at least one truncation criteria in podsize()')

    if energy_thresh is not None:
        svals_squared = np.square(svals.copy())
        energies = np.cumsum(svals_squared)
        energies /= np.square(svals).sum()
        numvecs = np.where(energies >= energy_thresh)[0][0]
    else:
        numvecs = min_size

    if min_size is not None and numvecs < min_size:
        numvecs = min_size

    if max_size is not None and numvecs > max_size:
        numvecs = max_size

    return numvecs

def compute_ECSW_training_matrix(snaps, prev_snaps, basis, res, jac, grid, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """
    n_hdm, n_snaps = snaps.shape
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))
    for isnap in range(1,n_snaps):
        snap = prev_snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        u_proj = (basis.dot(basis.T)).dot(snap)
        ires = res(snap, grid, dt, uprev, mu)
        Ji = jac(snap, grid, dt)
        Wi = Ji.dot(basis)
        rki = Wi.T.dot(ires)
        for inode in range(n_hdm):
            C[isnap*n_pod:isnap*n_pod+n_pod, inode] = ires[inode]*Wi[inode]

    return C

def compute_error(rom_snaps, hdm_snaps):
    """ Computes the relative error at each timestep """
    sq_hdm = np.sqrt(np.square(rom_snaps).sum(axis=0))
    sq_err = np.sqrt(np.square(rom_snaps - hdm_snaps).sum(axis=0))
    rel_err = sq_err / sq_hdm
    return rel_err, rel_err.mean()

def param_to_snap_fn(mu, snap_folder="param_snaps", suffix='.npy'):
    npar = len(mu)
    snapfn = snap_folder + '/'
    for i in range(npar):
        if i > 0:
            snapfn += '+'
        param_str = 'mu{}_{}'.format(i+1, mu[i])
        snapfn += param_str
    return snapfn + suffix

def get_saved_params(snap_folder="param_snaps"):
    param_fn_set = set(glob.glob(snap_folder+'/*'))
    return param_fn_set

def load_or_compute_snaps(mu, grid, w0, dt, num_steps, snap_folder="param_snaps"):
    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    saved_params = get_saved_params(snap_folder=snap_folder)
    if snap_fn in saved_params:
        print("Loading saved snaps for mu1={}, mu2={}".format(mu[0], mu[1]))
        snaps = np.load(snap_fn)[:, :num_steps+1]
    else:
        snaps = viscous_burgers_implicit(grid, w0, dt, num_steps, mu)
        np.save(snap_fn, snaps)
    return snaps

def plot_snaps(grid, snaps, snaps_to_plot, linewidth=2, color='black', linestyle='solid', 
               label=None, fig_ax=None):
    if (fig_ax is None):
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    x = (grid[1:] + grid[:-1])/2
    is_first_line = True
    for ind in snaps_to_plot:
        if is_first_line:
            label2 = label
            is_first_line = False
        else:
            label2 = None
        ax.plot(x, snaps[:,ind], 
                color=color, linestyle=linestyle, linewidth=linewidth, label=label2)

    return fig, ax
