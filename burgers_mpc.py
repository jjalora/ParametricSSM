import pickle
import time
from os.path import dirname, abspath, join
import sys
import numpy as np
import scipy
from scipy.optimize import nnls
from matplotlib import pyplot as plt
from matplotlib import cm
import argparse
import jax

from scp.models.burgers_1d import Burgers_1D, Burgers_1D_ROM, Burgers_1D_HROM
from scp.MPC import NonlinearMPCController
from scp.utils import HyperRectangle, QuadraticCost, save_data, load_data
from hypernet_viscous import make_1D_grid, plot_snaps
from burgers_utils import compute_ECSW_training_matrix
from scp.measurement import MeasurementModel
from scp.observer import RO_Observer, DiscreteEKFObserver

path = dirname(abspath(__file__))
root = dirname(path)
sys.path.append(root)

def main():
    parser = argparse.ArgumentParser(description="Runs MPC on Burgers 1D example")
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    # Full-order model
    dt = 0.01
    num_steps = 501
    num_cells = 350
    xl, xu = 0, 1
    mu = np.array([5e-3])
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2

    # Model parameters
    n_x = num_cells
    n_u = 5
    n_z = n_x
    model = Burgers_1D(n_x, n_u, n_z, mu, grid)

    # Cost function
    cost_params = QuadraticCost()
    cost_params.R = .001 * np.eye(model.n_u)
    cost_params.Q = 100 * np.eye(model.n_x)

    # Setpoint
    zf_target = np.zeros((n_x,))

    # Constraints
    X = None
    dU = None
    u_low, u_hi = -2, 2
    umin = np.array([u_low, u_low, u_low, u_low, u_low])
    umax = np.array([u_hi, u_hi, u_hi, u_hi, u_hi])
    U = HyperRectangle(umax, umin)


    # Initial condition
    x0 = 0.5 * np.sin(np.pi * xc) ** 2

    # Define target trajectory or setpoint
    planning_horizon = 3

    t0 = time.time()
    nmpc_controller_FOM = NonlinearMPCController(model=model, N=planning_horizon, dt=dt, Qz=cost_params.Q, R=cost_params.R,
                                                 z=zf_target, x0=x0, U=U, X=X, dU=dU,
                                            verbose=1, warm_start=True, solver='OSQP')

    w_cl, u_cl, t_span = model.simulate(x0, nmpc_controller_FOM, num_steps, dt)

    tf = time.time()
    print("---Solve Time---: {} ms".format(tf - t0))

    # Store results for use later
    snapshots_file = join(path, 'snapshots.pkl')
    snapshots = {}
    snapshots['u'] = u_cl
    snapshots['w'] = w_cl
    snapshots['t'] = t_span
    snapshots['x'] = xc
    # snapshots['x'] = np.tile(xc, (t_span.size, 1))

    save_data(snapshots_file, snapshots)

    # Test functions
    # dt = 0.01
    # num_steps = 500
    # num_cells = 50
    #
    # u =  0 * np.ones((num_steps, 5))
    # _, z = model.rollout(x0, u, dt)

    # Plot states
    fig, ax = plt.subplots()
    snaps_to_plot = range(50, 500, 50)
    fig, ax = plot_snaps(grid, w_cl.T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()

def loadAndPlotResults():

    # Load snapshot data
    snap_file = join(path, 'snapshots.pkl')
    snap_data = load_data(snap_file)
    u_snap = np.array(snap_data['u'])
    x_snap = snap_data['x']
    t_snap = snap_data['t']
    w_snap = snap_data['w']

    ##########################################
    # Plot temporal evolution of 1D Burger's
    ##########################################
    fig1 = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
    ax1 = fig1.add_subplot(111, projection='3d')
    t, grid = np.meshgrid(t_snap, x_snap)

    ax1.plot_surface(grid, t, w_snap.T, cmap=cm.coolwarm)

    spatialFig_file = join(path, 'burgers_x_vs_t.png')
    plt.savefig(spatialFig_file, dpi=300, bbox_inches='tight')

    ##########################################
    # Plot time series for input
    ##########################################
    fig2, axs2 = plt.subplots(5, 1, sharex='col')

    axs2[0].plot(t_snap[:-1], u_snap[:, 0], 'tab:green', label='Section 1 Input')
    axs2[1].plot(t_snap[:-1], u_snap[:, 1], 'tab:orange', label='Section 2 Input')
    axs2[2].plot(t_snap[:-1], u_snap[:, 2], 'tab:blue', label='Section 3 Input')
    axs2[3].plot(t_snap[:-1], u_snap[:, 3], 'tab:red', label='Section 4 Input')
    axs2[4].plot(t_snap[:-1], u_snap[:, 4], 'tab:pink', label='Section 5 Input')
    plt.show()

    ##########################################
    # Plot 1D evolution of Burger's
    ##########################################
    num_steps = len(t_snap)
    w_snap = w_snap[:, :-1]
    fig, ax = plt.subplots()
    snaps_to_plot = range(int(num_steps / 10), num_steps, int(num_steps / 10))
    fig, ax = plot_snaps(grid, w_snap.T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    plt.show()


def open_loop():
    parser = argparse.ArgumentParser(description="Runs MPC on Burgers 1D example")
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    # Full-order model
    dt = 0.01
    num_steps = 501
    num_cells = 350
    xl, xu = 0, 1
    mu = np.array([5e-3])
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2

    # Model parameters
    n_x = num_cells
    n_u = 5
    n_z = n_x
    model = Burgers_1D(n_x, n_u, n_z, mu, grid)

    # Initial condition
    x0 = 0.5 * np.sin(np.pi * xc) ** 2
    # x0 = 0.5 * np.sin(2 * np.pi * xc)

    ts = np.linspace(0, num_steps * dt - dt, num_steps)
    xs = np.zeros((num_steps, x0.shape[0]))
    us = [None] * (num_steps - 1)

    xs[0, :] = x0
    for j in range(num_steps - 1):
        print(" ... Solving timestep {}".format(j))
        x = xs[j, :]
        t = ts[j]

        # Simulate using built-in time-stepper
        xs[j + 1, :] = model.get_next_state(x, np.zeros(n_u,), dt)
        if j % 10 == 0:
            plt.plot(xs[j+1, :])
            plt.show()

    # Store results for use later
    snapshots = {}
    snapshots['w'] = xs
    snapshots['x'] = xc
    # snapshots['x'] = np.tile(xc, (t_span.size, 1))

    # Test functions
    # dt = 0.01
    # num_steps = 500
    # num_cells = 50
    #
    # u =  0 * np.ones((num_steps, 5))
    # _, z = model.rollout(x0, u, dt)

    # Plot states
    fig, ax = plt.subplots()
    snaps_to_plot = range(50, 500, 50)
    fig, ax = plot_snaps(grid, xs.T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()

def open_loop_rom():
    parser = argparse.ArgumentParser(description="Runs MPC on Burgers 1D example")
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    with open(r'snapshots_350.pkl', 'rb') as fh:
        data = pickle.load(fh)

    Vfull, s, _ = scipy.linalg.svd(data['w'].T, full_matrices=False)

    # Full-order model
    dt = 0.01
    num_steps = 501
    num_cells = 350
    xl, xu = 0, 1
    mu = np.array([5e-3])
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2

    # Model parameters
    n_x = num_cells
    n_u = 5
    n_z = n_x
    # size of model
    n = 50
    model = Burgers_1D_ROM(n_x, n_u, n_z, mu, grid, Vfull[:, :n])

    # Initial condition
    x0 = 0.5 * np.sin(np.pi * xc) ** 2

    ts = np.linspace(0, num_steps * dt - dt, num_steps)
    xs = np.zeros((num_steps, x0.shape[0]))
    us = [None] * (num_steps - 1)

    xs[0, :] = x0
    for j in range(num_steps - 1):
        print(" ... Solving timestep {}".format(j))
        x = xs[j, :]
        t = ts[j]

        # Simulate using built-in time-stepper
        xs[j + 1, :] = model.get_next_state(x, np.zeros(1,), dt)

    # Store results for use later
    snapshots = {}
    snapshots['w'] = xs
    snapshots['x'] = xc
    # snapshots['x'] = np.tile(xc, (t_span.size, 1))

    # Test functions
    # dt = 0.01
    # num_steps = 500
    # num_cells = 50
    #
    # u =  0 * np.ones((num_steps, 5))
    # _, z = model.rollout(x0, u, dt)

    # Plot states
    fig, ax = plt.subplots()
    snaps_to_plot = range(50, 500, 50)
    fig, ax = plot_snaps(grid, xs.T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()

def open_loop_fom():
    # Full-order model
    dt = 0.01
    num_steps = 501
    num_cells = 350
    xl, xu = 0, 1
    mu = np.array([5e-3])
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2

    # Model parameters
    n_x = num_cells
    n_u = 5
    n_z = n_x
    # size of model
    n = 40

    model = Burgers_1D(n_x, n_u, n_z, mu, grid)

    # Initial condition
    x0 = 0.5 * np.sin(np.pi * xc) ** 2

    ts = np.linspace(0, num_steps * dt - dt, num_steps)
    xs = np.zeros((num_steps, x0.shape[0]))
    us = [None] * (num_steps - 1)

    xs[0, :] = x0
    for j in range(num_steps - 1):
        print(" ... Solving timestep {}".format(j))
        x = xs[j, :]
        t = ts[j]

        # Simulate using built-in time-stepper
        xs[j + 1, :] = model.get_next_state(x, np.zeros(n_u,), dt)

    # Store results for use later
    snapshots = {}
    snapshots['w'] = xs
    snapshots['x'] = xc

    snapshots_file = join(path, 'snapshots_open_loop.pkl')
    save_data(snapshots_file, snapshots)

    # Plot states
    fig, ax = plt.subplots()
    snaps_to_plot = range(50, num_steps - 1, 50)
    fig, ax = plot_snaps(grid, snapshots['w'].T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='HPROM')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('HDM')
    ax.legend()
    plt.show()

def open_loop_hrom():
    parser = argparse.ArgumentParser(description="Runs MPC on Burgers 1D example")
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    file_name = r'snapshots_open_loop.pkl'
    # file_name = r'snapshots_350.pkl'
    with open(file_name, 'rb') as fh:
        data = pickle.load(fh)

    Vfull, s, _ = scipy.linalg.svd(data['w'].T, full_matrices=False)
    n_99 = np.argmin(np.abs(np.cumsum(s)/np.sum(s) - 0.9999))
    print(f'Size of basis for 99.99% energy retention: {n_99}')

    # Full-order model
    dt = 0.01
    num_steps = 501
    num_cells = 350
    xl, xu = 0, 1
    mu = np.array([5e-3])
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2

    # Model parameters
    n_x = num_cells
    n_u = 5
    n_z = n_x
    # size of model
    n = 10

    snaps = data['w'][0:50:4, :].T
    snapsp = data['w'][3:53:4, :].T
    # plt.plot(snaps)
    # plt.plot(snapsp)
    # plt.show()
    # plt.plot([np.linalg.norm(v) for v in snaps.T])
    # plt.show()
    C = compute_ECSW_training_matrix(snaps, snapsp, Vfull[:, :n], grid, dt, mu)
    # C = C.numpy()

    buffer = 5
    C = C[:, buffer:(-buffer)]
    interior, _ = nnls(C, C.sum(axis=1), maxiter=9999999)
    weights = np.ones(snaps[:, 0].shape)
    weights[buffer:(-buffer)] = interior
    # plt.clf()
    # plt.spy(interior)

    print('sum(weights): {}'.format(weights.sum()))
    print('nnz(weights) / weights.shape: {} / {}'.format((weights > 0).sum(), weights.shape[0]))

    model = Burgers_1D_HROM(n_x, n_u, n_z, mu, grid, Vfull[:, :n], weights)

    # Initial condition
    x0 = Vfull[:, :n].T @ (0.5 * np.sin(np.pi * xc) ** 2)

    ts = np.linspace(0, num_steps * dt - dt, num_steps)
    xs = np.zeros((num_steps, x0.shape[0]))
    us = [None] * (num_steps - 1)

    xs[0, :] = x0
    for j in range(num_steps - 1):
        print(" ... Solving timestep {}".format(j))
        x = xs[j, :]
        t = ts[j]

        # Simulate using built-in time-stepper
        xs[j + 1, :] = model.get_next_state(x, np.zeros(n_u,), dt)

    # Store results for use later
    snapshots = {}
    snapshots['w'] = (Vfull[:, :n] @ xs.T).T
    snapshots['x'] = xc
    # snapshots['x'] = np.tile(xc, (t_span.size, 1))

    # Test functions
    # dt = 0.01
    # num_steps = 500
    # num_cells = 50
    #
    # u =  0 * np.ones((num_steps, 5))
    # _, z = model.rollout(x0, u, dt)

    # Plot states
    fig, ax = plt.subplots()
    snaps_to_plot = range(50, num_steps - 1, 50)
    fig, ax = plot_snaps(grid, snapshots['w'].T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='HPROM')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('HPROM (n = {})'.format(n))
    ax.legend()
    plt.show()

def closed_loop_hrom():
    parser = argparse.ArgumentParser(description="Runs MPC on Burgers 1D example")
    parser.add_argument('--output_folder', type=str, default=None)
    args = parser.parse_args()

    with open(r'snapshots_350.pkl', 'rb') as fh:
        data = pickle.load(fh)

    Vfull, s, _ = scipy.linalg.svd(data['w'].T, full_matrices=False)

    # Full-order model
    dt = 0.01
    num_steps = 200
    num_cells = 350
    xl, xu = 0, 1
    mu = np.array([5e-3])
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2

    # Nodes to observe
    sample_node = 50
    nodes = np.arange(0, num_cells, sample_node)

    # Model parameters
    n = 10
    n_x = n
    n_u = 5
    n_z = len(nodes)
    
    # Construct measurement model
    outputModel = MeasurementModel(nodes, num_cells) # TODO: No noise

    snaps = data['w'][0:-1:20, :].T
    snapsp = data['w'][3:-1:20, :].T
    C = compute_ECSW_training_matrix(snaps, snapsp, Vfull[:, :n], grid, dt, mu)
    # C = C.numpy()

    buffer = 5
    C = C[:, buffer:(-buffer)]
    interior, _ = nnls(C, C.sum(axis=1), maxiter=9999999)
    weights = np.ones(snaps[:, 0].shape)
    weights[buffer:(-buffer)] = interior
    # print('WARNING: SAMPLING ALL NODES REGARDLESS OF ECSW REDUCED MESH')
    print('sum(weights): {}'.format(weights.sum()))
    print('nnz(weights) / weights.shape: {} / {}'.format((weights > 0).sum(), weights.shape[0]))

    # Construct simulation and control model
    model_hdm = Burgers_1D(num_cells, n_u, len(nodes), mu, grid, outputModel)
    model = Burgers_1D_HROM(n_x, n_u, n_z, mu, grid, Vfull[:, :n], weights, outputModel)
    
    # Construct observer model
    # observerModel = RO_Observer(model)
    observerModel = DiscreteEKFObserver(model)

    # Cost function
    cost_params = QuadraticCost()
    cost_params.R = .001 * np.eye(model.n_u)
    cost_params.Q = 100 * np.eye(model.n_z)

    # Setpoint
    # zf_target = Vfull[:, :n].T @ np.zeros((num_cells,))
    zf_target = np.zeros((n_z,))

    # Constraints
    X = None
    dU = None
    u_low, u_hi = -2, 2
    umin = np.array([u_low, u_low, u_low, u_low, u_low])
    umax = np.array([u_hi, u_hi, u_hi, u_hi, u_hi])
    U = HyperRectangle(umax, umin)


    # Initial condition
    x0 = Vfull[:, :n].T@(0.5 * np.sin(np.pi * xc) ** 2)
    x0_hdm = (0.5 * np.sin(np.pi * xc) ** 2)

    # Define target trajectory or setpoint
    planning_horizon = 3

    t0 = time.time()
    nmpc_controller_FOM = NonlinearMPCController(model=model, observer=observerModel, N=planning_horizon, dt=dt, Qz=cost_params.Q, R=cost_params.R,
                                                 z=zf_target, x0=x0, U=U, X=X, dU=dU,
                                            verbose=1, warm_start=True, solver='OSQP')

    w_cl, u_cl, t_span = model_hdm.simulate(x0_hdm, nmpc_controller_FOM, num_steps, dt)
    print('w_cl.shape', w_cl.shape)

    tf = time.time()
    print("---Solve Time---: {} s".format(tf - t0))

    # Store results for use later
    snapshots_file = join(path, 'snapshots.pkl')
    snapshots = {}
    snapshots['u'] = u_cl
    snapshots['w'] = w_cl
    snapshots['t'] = t_span
    snapshots['x'] = xc
    snapshots['V'] = Vfull[:, :n]
    # snapshots['x'] = np.tile(xc, (t_span.size, 1))

    save_data(snapshots_file, snapshots)

    # Test functions
    # dt = 0.01
    # num_steps = 500
    # num_cells = 50
    #
    # u =  0 * np.ones((num_steps, 5))
    # _, z = model.rollout(x0, u, dt)

    # Plot states
    fig, ax = plt.subplots()
    snaps_to_plot = range(int(num_steps / 10), num_steps, int(num_steps / 10))
    fig, ax = plot_snaps(grid, w_cl.T, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    # main()
    loadAndPlotResults()
    # open_loop()
    # open_loop_rom()
    # open_loop_hrom()
    # closed_loop_hrom()
    # open_loop_fom()