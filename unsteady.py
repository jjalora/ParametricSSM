"""
Run the HDM, comparing implicit and explicit methods to debug. 
"""
import sys
import time

import jax
import numpy as np
import matplotlib.pyplot as plt
from hypernet_viscous import viscous_burgers_explicit, viscous_burgers_implicit, make_1D_grid, plot_snaps
from burgers_utils import *

def main():
    # first do an explicit solve with a small timestep, as a debugging reference
    dt = 0.0001
    num_steps = 50000
    num_cells = 500
    xl, xu = 0, 1
    mu = np.array([5e-3])

    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1])/2
    w0 = 0.5*np.sin(np.pi*xc)**2
    # w0[grid > 0.5] = 0
    snaps = viscous_burgers_explicit(grid, w0, dt, num_steps, mu)

    fig, ax = plt.subplots()
    snaps_to_plot = range(5000, 50001, 5000)
    fig, ax = plot_snaps(grid, snaps, snaps_to_plot, fig_ax=(fig, ax),
                         linewidth=2, label='explicit')

    # compare implicit solve
    dt = 0.01
    num_steps = 500
    num_cells = 500
    xl, xu = 0, 1
    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2
    w0 = 0.5 * np.sin(np.pi * xc) ** 2
    snaps = viscous_burgers_implicit(grid, w0, dt, num_steps, mu)

    snaps_to_plot = range(50, 501, 50)
    fig, ax = plot_snaps(grid, snaps, snaps_to_plot, fig_ax=(fig,ax), 
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()

def run_controller():
    # compare implicit solve
    dt = 0.01
    num_steps = 500
    num_cells = 50
    xl, xu = 0, 1
    mu = np.array([5e-3])

    grid = make_1D_grid(xl, xu, num_cells)
    xc = (grid[1:] + grid[:-1]) / 2
    w0 = 0.5 * np.sin(np.pi * xc) ** 2
    u = 0 * np.ones((num_steps, 5))

    snaps = viscous_burgers_implicit_jit(grid, w0, dt, num_steps, mu)

    fig, ax = plt.subplots()
    snaps_to_plot = range(50, 501, 50)
    fig, ax = plot_snaps(grid, snaps, snaps_to_plot, fig_ax=(fig, ax),
                         linestyle='dashed', color='red', label='implicit')
    ax.set_xlim([grid.min(), grid.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.set_title('Burgers Equation Snapshots')
    ax.legend()
    plt.show()

    # Try to autodiff for jacobian then modify sparse Jacobian and compare
    # A = jax.jacobian(viscous_burgers_implicit_step, (1,))(grid, w0, dt, mu)
    # print('Autodiffed Jacobian is: ', A)



if __name__ == "__main__":
    if sys.argv[1] == 'control':
        run_controller()
    else:
        main()
