import time

import numpy as np
from matplotlib import pyplot as plt

from scp.locp import LOCP

#### Default variables for GuSTO ####
DELTA0 = 1e4  # trust region
OMEGA0 = 1  # slack variable weighting
RHO = 0.1  # model compute_accuracy
# RHO0 = 0.005 # no longer needed
BETA_FAIL = 0.5
BETA_SUCC = 2
EPSILON = 0.01
GAMMA_FAIL = 5
OMEGA_MAX = 1e10
MAX_ITERS = 500
CONVERGE = 0.1


class GuSTO:
    """
    GuSTO class for solving trajectory optimization problems via SQP

    :model: TemplateModel object describing dynamics (see scp/models/template.py)
    :N: integer optimization horizon
    :dt: time step (seconds)
    :Qz: positive semi-definite performance variable weighting matrix (n_z, n_z)
    :R: positive definite control weighting matrix (n_u, n_u)
    :x0: initial condition (n_x,)
    :u_init: control initial guess (N, n_u)
    :x_init: state initial guess (N+1, n_x)
    :z: (optional) desired tracking trajectory for objective function (N+1, n_z)
    :u: (optional) desired control for objective function (N, n_u)
    :Qzf: (optional) positive semi-definite terminal performance variable weighting matrix (n_z, n_z)
    :zf: (optional) terminal target state (n_z,), defaults to 0 if Qzf provided
    :U: (optional) control constraint (Polyhedron object)
    :X: (optional) state constraint (Polyhedron object)
    :Xf: (optional) terminalstate constraint (Polyhedron object)
    :dU: (optional) u_k - u_{k-1} constraint Polyhedron object
    :verbose: (optional) 0,1,2 varying levels of verbosity (default 0)
    :visual: (optional) list of indices of z to plot at each iteration
    :warm_start: (optional) boolean (default True)
    :x_char: (optional) characteristic quantities for x, for scaling
    :f_char: (optional) characteristic quantities for f, for scaling
    :kwargs: Keyword arguments for GuSTO (see below) and optionally for the solver
    (https://osqp.org/docs/interfaces/solver_settings.html)
    """

    def __init__(self, model, N, dt, Qz, R, x0, u_init, x_init, z=None, u=None,
                 Qzf=None, zf=None, U=None, X=None, Xf=None, dU=None,
                 verbose=0, visual=None, warm_start=True, periodic=False, t=None, **kwargs):
        self.model = model

        self.n_x = x0.shape[0]
        self.n_u = R.shape[0]
        self.n_z = Qz.shape[0]
        self.dt = dt
        self.N = N  # time horizon

        #### Cost matrix values ####
        self.Qz = Qz
        self.R = R
        self.Qzf = Qzf

        #### Constraints - State, control, final state ####
        self.U = U
        self.X = X
        self.Xf = Xf
        self.dU = dU
        self.periodic = periodic
        self.t = t

        self.verbose = verbose
        self.visual = visual

        self.locp_solve_time = None

        #### Parameters ####
        ## Initial cost parameter values ##
        self.delta0 = kwargs.get('delta0', DELTA0)
        # pop is used to remove these args from kwargs, which is passed to cvxpy solver, else invalid keyword argument
        kwargs.pop('delta0', None)
        self.omega0 = kwargs.get('omega0', OMEGA0)
        kwargs.pop('omega0', None)

        ## Model accuracy parameters ##
        self.rho = kwargs.get('rho', RHO)
        kwargs.pop('rho', None)
        # self.rho0 = kwargs.get('rho0', RHO0)

        ## Trust region parameters ##
        self.beta_fail = kwargs.get('beta_fail', BETA_FAIL)
        kwargs.pop('beta_fail', None)
        self.beta_succ = kwargs.get('beta_succ', BETA_SUCC)
        kwargs.pop('beta_succ', None)

        ## Cost function penalty term parameters ##
        self.gamma_fail = kwargs.get('gamma_fail', GAMMA_FAIL)
        kwargs.pop('gamma_fail', None)
        self.omega_max = kwargs.get('omega_max', OMEGA_MAX)
        kwargs.pop('omega_max', None)

        ## Constraint violation parameters ##
        self.epsilon = kwargs.get('epsilon', EPSILON)
        kwargs.pop('epsilon', None)

        ## Convergence threshold ##
        self.convg_thresh = kwargs.get('convg_thresh', CONVERGE)
        kwargs.pop('convg_thresh', None)

        ## Characteristic quantities ##
        self.x_char = kwargs.get('x_char', np.ones(self.n_x))
        self.x_scale = 1. / np.abs(self.x_char)
        kwargs.pop('x_char', None)
        self.f_char = kwargs.get('f_char', np.ones(self.n_x))
        self.f_scale = 1. / np.abs(self.f_char)
        kwargs.pop('f_char', None)

        ## Problem parameters ##
        self.x_k = None  # Previous state
        self.u_k = None  # Previous input

        # LOCP problem
        if self.verbose == 2:
            locp_verbose = True
        else:
            locp_verbose = False
        self.locp = LOCP(self.N, self.model.H, self.Qz, self.R, Qzf=self.Qzf,
                         U=self.U, X=self.X, Xf=self.Xf, dU=self.dU,
                         verbose=locp_verbose, warm_start=warm_start, x_char=self.x_char, periodic=self.periodic, **kwargs)

        # Solve SCP
        self.max_gusto_iters = MAX_ITERS # let first solve take more time
        self.solve(x0, u_init, x_init, z, zf, u)

        ## Problem parameters ##
        self.max_gusto_iters = kwargs.get('max_gusto_iters', MAX_ITERS)
        kwargs.pop('max_gusto_iters', None)


    def is_converged(self, x, u):
        """
        Sequential problem is converged when current state input pair is same as previous state input pair
        """
        dx = (1. / self.n_x) * np.sum(np.linalg.norm(np.multiply(self.x_scale, x - self.x_k), axis=1))
        # du = (1./self.n_u) * np.sum(np.linalg.norm(u - self.u_k, axis=1))
        # dsol = (1./self.N) * (dx + du)
        dsol = (1. / self.N) * dx
        if dsol <= self.convg_thresh:
            return dsol, True
        else:
            return dsol, False

    def is_valid_iteration(self, itr):
        """
        Is the current iteration within the limits
        :param itr: Iteration of sequential convex program
        :return: Boolean
        """
        if itr <= self.max_gusto_iters:
            return True
        else:
            return False

    def is_in_trust_region(self, x, delta):
        """
        :param itr: Iteration of Sequential solve
        Returns True if the solution lies in the trust region w.r.t. previous trust region
        """
        max_diff = np.max(np.linalg.norm(np.multiply(self.x_scale, x - self.x_k), np.inf, axis=1))
        if max_diff - delta > self.epsilon:
            return max_diff, False
        else:
            return 0.0, True

    def state_constraints_violated(self, x):
        """
        For GuSTO state constraints get enforced as penalties, not as strict constraints. Computes whether the state
        constraints are within a user-chosen tolerance epsilon
        Requires both state and final state constraints to be specified by Polyhedra (see gusto_utils.py)
        """
        max_violation = 0.0
        if self.X is not None:
            for i in range(x.shape[0]):
                val = self.X.get_constraint_violation(x[i, :])
                if val > max_violation:
                    max_violation = val

        if max_violation > self.epsilon:
            return max_violation, False
        else:
            return max_violation, True

    def compute_accuracy(self, x, u, J):
        """
        Assumes cost function is quadratic and cost function approximation is hence exact --> 0 error
        Computes model accuracy of dynamics
        """
        error = 0
        approx = 0
        for i in range(x.shape[0] - 1):
            # Get true dynamics at the current points
            if self.t is not None:
                fk, Ak, Bk = self.model.get_continuous_dynamics(self.x_k[i, :], self.u_k[i, :], t=self.t[i])
            else:
                fk, Ak, Bk = self.model.get_continuous_dynamics(self.x_k[i, :], self.u_k[i, :])

            # Get dynamics at the potential new solution
            if self.t is not None:
                f, _, _ = self.model.get_continuous_dynamics(x[i, :], u[i, :], t=self.t[i])
            else:
                f, _, _ = self.model.get_continuous_dynamics(x[i, :], u[i, :])

            # Compute approximation of f(x,u) via Taylor expansion about (xk, uk)
            f_approx = fk + Ak @ (x[i, :] - self.x_k[i, :]) + Bk @ (u[i, :] - self.u_k[i, :])
            error += self.dt * np.linalg.norm(np.multiply(self.f_scale, f - f_approx), 2)
            approx += self.dt * np.linalg.norm(np.multiply(self.f_scale, f_approx), 2)

        rho_k = error / (J + approx)
        return rho_k

    def get_traj_dynamics(self, x, u):
        """
        Return the affine dynamics of each point along trajectory in a list
        """
        A_d = []
        B_d = []
        d_d = []
        for i in range(x.shape[0] - 1):
            if self.t is not None:
                A_d_i, B_d_i, d_d_i = self.model.get_discrete_dynamics(x[i, :], u[i, :], self.dt, t=self.t[i])
            else:
                A_d_i, B_d_i, d_d_i = self.model.get_discrete_dynamics(x[i, :], u[i, :], self.dt)
            A_d.append(A_d_i)
            B_d.append(B_d_i)
            d_d.append(d_d_i)

        return A_d, B_d, d_d

    def solve(self, x0, u_init, x_init, z=None, zf=None, u=None):
        """
        :x0: initial condition np.array
        :u_init: control initial guess (N, n_u)
        :x_init: state initial guess (N+1, n_x)
        :z: (optional) desired tracking trajectory for objective function (N+1, n_z)
        :zf: (optional) desired terminal state for objective function (n_z,)
        :u: (optional) desired control for objective function (N, n_z)
        """
        # Timing information to be stored
        t0 = time.time()
        t_locp = 0.0

        itr = 0
        self.u_k = u_init
        self.x_k = x_init
        A_d, B_d, d_d = self.get_traj_dynamics(self.x_k, self.u_k)
        new_solution = True
        Jstar_prev = np.inf
        delta_prev = np.inf
        omega_prev = np.inf

        converged = False

        delta = self.delta0
        omega = self.omega0

        if self.verbose >= 1:
            print('|   J   | TR_viol |  rho_k  |  X_viol |   x-x_k |  delta  |  omega |')
            print('--------------------------------------------------------------------')

        while self.is_valid_iteration(itr) and not converged and omega <= self.omega_max:
            rho_k = -1
            max_violation = -1
            dsol = -1
            delta_cur = delta  # just for printing
            omega_cur = omega  # just for printing

            # Update the LOCP with new parameters and solve
            if new_solution:
                self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u)
                new_solution = False
            else:
                self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, full=False)

            # Solve the LOCP
            Jstar, success, stats = self.locp.solve()
            if not success:
                print('Iteration {} of problem cannot be solved, see solver status for more information'.format(itr))
                self.xopt = np.copy(self.x_k)
                self.uopt = np.copy(self.u_k)
                self.zopt = np.transpose(self.model.H @ self.xopt.T)
                return

            t_locp += stats.solve_time
            x_next, u_next, _ = self.locp.get_solution()

            # Check if trust region is satisfied
            e_tr, tr_satisfied = self.is_in_trust_region(x_next, delta)
            if tr_satisfied:
                rho_k = self.compute_accuracy(x_next, u_next, Jstar)

                if rho_k > self.rho:
                    delta = self.beta_fail * delta
                else:
                    """
                    First modification to GuSTO: if delta and omega are constant for two solves in a row,
                    yet the reported cost of the optimizer increases, decrease delta
                    """
                    if delta_prev == delta and omega_prev == omega and Jstar_prev <= Jstar:
                        delta = self.beta_fail * delta
                    delta_prev = delta
                    Jstar_prev = Jstar
                    omega_prev = omega

                    """
                    Second modification to GuSTO: remove delta increases for good model accuracy
                    """
                    # if rho_k < self.rho0:
                    #     delta = np.minimum(self.beta_succ * delta, self.delta0)
                    # else:
                    #     delta = delta

                    # Computes g2
                    max_violation, X_satisfied = self.state_constraints_violated(x_next)

                    """
                    Third modification to GuSTO: remove decreases of omega for satisifed X (creates oscillations)
                    """
                    # if X_satisfied:
                    #     omega = self.omega0
                    # else:
                    #     omega = self.gamma_fail * omega
                    if not X_satisfied:
                        omega = self.gamma_fail * omega

                    # Check for convergence
                    dsol, converged = self.is_converged(x_next, u_next)

                    # Optional: Enforce state constraints are satisfied upon convergence
                    if not X_satisfied:
                        converged = False

                    # Record that a new solution as been found
                    new_solution = True

            else:
                omega = self.gamma_fail * omega

            itr += 1

            if self.verbose >= 1:
                if rho_k < 0.0:
                    print('{:.2e}, {:.2e}, {}, {}, {}, {:.2e}, {:.2e}, {}'.format(
                        Jstar, e_tr, '-' * 8, '-' * 8, '-' * 8, delta_cur, omega_cur, itr))
                elif max_violation < 0.0:
                    print('{:.2e}, {:.2e}, {:.2e}, {}, {}, {:.2e}, {:.2e}, {}'.format(
                        Jstar, e_tr, rho_k, '-' * 8, '-' * 8, delta_cur, omega_cur, itr))
                else:
                    print('{:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {}'.format(
                        Jstar, e_tr, rho_k, max_violation, dsol, delta_cur, omega_cur, itr))

            # Plot solution
            if self.visual:
                z_k = self.model.H @ self.x_k.T
                z_new = self.model.H @ x_next.T
                for i in self.visual:
                    plt.plot(z_k[i], 'b--')
                    plt.plot(z_new[i], 'b')
                plt.title('--: old, -: new, accepted: {}'.format(new_solution))
                plt.show()

            # If valid solution, update and recompute dynamics
            if new_solution:
                self.x_k = x_next.copy()
                self.u_k = u_next.copy()
                A_d, B_d, d_d = self.get_traj_dynamics(self.x_k, self.u_k)

        t_gusto = time.time() - t0
        if omega > self.omega_max:
            print('omega > omega_max, solution did not converge')
        if not self.is_valid_iteration(itr-1):
            print('Max iterations, solution did not converge')
        else:
            print('Solved in {} iterations/{:.3f} seconds, with {:.3f} s from LOCP solve'.format(itr, t_gusto, t_locp))

        # Save optimal solution
        self.xopt = np.copy(self.x_k)
        self.uopt = np.copy(self.u_k)
        self.zopt = np.transpose(self.model.H @ self.xopt.T)
        self.locp_solve_time = t_locp

    def get_solution(self):
        return self.xopt, self.uopt, self.zopt, self.locp_solve_time
