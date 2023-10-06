import os

import numpy as np
from florianworld.utils import convert_matlab_to_numpy
from scipy.integrate import solve_ivp
from sofacontrol.scp.models.gen.vehicle_model import vehicle_model
from sofacontrol.scp.models.gen.vehicle_model_Amat import vehicle_model_Amat
from sofacontrol.scp.models.gen.vehicle_model_Bmat import vehicle_model_Bmat

class DoubleTrackVehicle:
    """
    Double track vehicle model object for use with GuSTO class. This object describes continuous time dynamics:

    xdot = f(x,u) = f0(x) + B(x)u
    z = Hx
    """

    def __init__(self):
        #### Dimensions of problem ####
        self.n_x = 11  # State dimension
        self.n_u = 3  # Input dimension
        self.n_z = 11  # Number of performance variables

        #### Objective function variables ####
        self.H = np.eye(self.n_x)  # Performance variable matrix (n_z x n_x)

    def get_continuous_dynamics(self, x, u, fonly=False):
        """
        For dynamics xdot = f(x,u) = f0(x) + B(x)u returns:
        f = f(x,u): full dynamics
        A = df/dx(x,u): state Jacobian (note contains f0(x) and B(x) terms)
        B = B(x): control Jacobian
        """
        RuntimeError('Must be subclassed and implemented')

        f = vehicle_model(x, u[0], u[1], u[2]).flatten()

        # Linearization

        # TODO: Singularity at (v_y, psi_dot) = (0, 0)
        if np.abs(x[2]) < 0.001:
            x[2] = 0.001

        A = vehicle_model_Amat(x, u[0], u[1], u[2]).T

        B = vehicle_model_Bmat(x)

        if fonly:
            return f
        else:
            return f, A, B

    def get_discrete_dynamics(self, x, u, dt):
        """
        For dynamics xdot = f0(x) + B(x)u, the Taylor approximation can be written

            xdot = f0(x0) + A(x0)(x-x0) + B(x0)u

        alternatively written as

            xdot =  A(x0)x + B(x0)u + d(x0)

        with d(x0) = f0(x0) - A(x0)x0 = f(x0,u0) - A(x0)x0 - B(x0)u0. This function
        then returns a discrete time version of this equation

            x_k+1 =  Ad x_k + Bd u_k + dd

        :x: State x0 (n_x)
        :u: Input u0 (n_u)
        :dt: time step for discretization (seconds)
        """
        f, A, B = self.get_continuous_dynamics(x, u)
        d = f - A @ x - B @ u

        # TODO: Forward Euler Implementation
        # dd = dt * d
        # Ad = np.eye(A.shape[0]) + dt * A
        # Bd = dt * B

        # TODO: Backward Euler
        Ad = np.linalg.inv(np.eye(A.shape[0]) - dt * A)
        sep_term = np.linalg.pinv(A) @ (Ad - np.eye(A.shape[0]))
        Bd = sep_term @ B
        dd = sep_term @ d

        # TODO: ZOH Discretization
        # Ad, Bd, dd = scutils.zoh_affine(A, B, d, dt)

        return Ad, Bd, dd

    def get_characteristic_vals(self):
        """
        An optional function to define a procedure for computing characteristic values
        of the state and dynamics for use with GuSTO scaling, defaults to all ones
        """
        x_char = np.ones(self.n_x)
        f_char = np.ones(self.n_x)
        return x_char, f_char

    def get_next_state(self, x, u, dt, stiff=False):
        f = self.get_continuous_dynamics(x, u, fonly=True)
        if not stiff:
            res = x + dt * f
        else:
            from scipy.optimize import fsolve
            residual = lambda xp, self, x, u, dt, fp: \
                xp - x - dt * fp(xp, u, fonly=True)
            # xp = x + dt * f
            res = fsolve(residual, x, args=(self, x, u, dt, self.get_continuous_dynamics))
        return res

    def rollout(self, x0, u, dt, stiff=False):
        """
        :x0: initial condition (n_x,)
        :u: array of control (N, n_u)
        :dt: time step

        Returns state x (N + 1, n_x) and performance variable z (N + 1, n_z)
        """
        N = u.shape[0]
        x = np.zeros((N + 1, self.n_x))

        # Set initial condition
        x[0, :] = x0

        # Simulate using some method
        for i in range(N):
            x[i + 1, :] = np.reshape(self.get_next_state(x[i, :], u[i, :], dt, stiff=stiff), -1)

        # Compute output variables
        if self.H is not None:
            z = np.transpose(self.H @ x.T)
        else:
            z = None

        RuntimeError('Must be subclassed and implemented')
        return x, z

    def step(self, x_0, u_0, dt, t_0, t_f, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with constant action over a
        time interval.

        Approximated using Runge-Kutta 4,5 solver.

        Inputs:
        Initial state, x_0: numpy array
        Control action, u_0: numpy array
        Time step, dt : float
        Initial time, t_0: float
        Final time, t_f: float
        Absolute tolerance, atol: float
        Relative tolerance, rtol: float

        Outputs:
        State at final time: numpy array
        """

        x_dot = lambda t, x: self.get_continuous_dynamics(x, u_0, fonly=True)
        t_span = [t_0, t_f]
        res = solve_ivp(x_dot, t_span, x_0, atol=atol, rtol=rtol, method='LSODA')
        return res.y[:, -1]

    def simulate(self, x0, controller, N, dt, x2p=None, atol=1e-6, rtol=1e-6):
        ts = np.linspace(0, N*dt, N+1)
        xs = np.zeros((N, self.n_x))
        us = [None] * (N - 1)

        xs[0, :] = x0
        for j in range(N - 1):
            x = xs[j, :]
            t = ts[j]
            if x2p is not None:
                topt, xopt, uopt, zopt, solve_time = controller.eval(x2p(x)[0], t)
            else:
                topt, xopt, uopt, zopt, solve_time = controller.eval(x, t)

            us[j] = uopt[0]
            # Simulate using RK45
            xs[j + 1, :] = self.step(x, uopt[0], dt, t, ts[j + 1], atol=atol, rtol=rtol)

            # Testing
            #xs[j + 1, :] = self.get_next_state(xopt[0], uopt[0], dt)

        return xs, us


def main():
    models_dir = os.path.dirname(__file__)
    matlab_dir = os.path.normpath(os.path.join(models_dir, "../../../../matlab/"))
    output_dir = os.path.normpath(os.path.join(models_dir, "gen"))
    vehicle_model_matlab = os.path.join(matlab_dir, "vehicle/gen/vehicle_model.m")
    vehicle_model_jacobian_matlab_A = os.path.join(matlab_dir, "vehicle/gen/vehicle_model_Amat.m")
    vehicle_model_jacobian_matlab_B = os.path.join(matlab_dir, "vehicle/gen/vehicle_model_Bmat.m")

    convert_matlab_to_numpy(vehicle_model_matlab, output_dir)
    convert_matlab_to_numpy(vehicle_model_jacobian_matlab_A, output_dir)
    convert_matlab_to_numpy(vehicle_model_jacobian_matlab_B, output_dir)

    import gen.vehicle_model as vm
    import gen.vehicle_model_jacobian as vmj
    import gen.vehicle_model_Amat as vmj_A
    import gen.vehicle_model_Bmat as vmj_B

    x0 = np.array([5.92310509838939e+000, -371.427872871926e-003, 207.946449170977e-003, 17.1946371132730e+000, 0.0,
                   18.2055263967147e+000, 18.3039941960485e+000, 17.2899492523066e+000, 0.0, 0.0, 0.0])

    u_a_ss = 100
    u_b_ss = 0
    u_s_ss = np.deg2rad(5)
    print(vm.vehicle_model(x0, u_a_ss, u_b_ss, u_s_ss))
    print(vmj_A.vehicle_model_Amat(x0, u_a_ss, u_b_ss, u_s_ss))
    print(vmj_B.vehicle_model_Bmat(x0))


if __name__ == '__main__':
    main()

