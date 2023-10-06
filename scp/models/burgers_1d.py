import jax
import numpy as np
import scp.utils as scutils
from scp.MPC import NonlinearMPCController
import jax.numpy as jnp
from burgers_utils import viscous_burgers_implicit_step, viscous_burgers_implicit_step_rom, \
    viscous_burgers_implicit_step_hrom, viscous_burgers_res_jax_hrom
from functools import partial

@jax.jit
def gudonov_flux(u_L, u, u_R):
    return 0.5*jnp.where(u > 0, u**2 - u_L**2, u_R**2 - u**2)

class Burgers_1D:
    """
    Burgers_1D model object for use with GuSTO class. This object describes continuous time dynamics:

    xdot = f(x,u) = f0(x) + B(x)u
    z = Hx
    """

    def __init__(self, n_x, n_u, n_z, mu, grid, measurement_model):
        #### Dimensions of problem ####
        self.n_x = n_x                              # State dimension
        self.n_u = n_u                              # Input dimension
        self.measurement_model = measurement_model  # Measurement model object
        self.n_z = n_z                              # Number of performance variables
        self.grid = grid                            # Discretized grid
        self.dx = self.grid[1] - self.grid[0]       # Resolution of grid (assume uniform)
        self.mu = mu                                # Diffusion coefficient

        # Construct control matrix
        self.B = np.zeros((n_x, n_u))
        interval = int(self.n_x / self.n_u)
        self.B[:interval, 0] = np.ones(interval)
        cntrl_idxs = np.arange(0, self.n_u)[1:]
        for i in cntrl_idxs:
            self.B[i*interval:(i+1)*interval, i] = np.ones(interval)

    def get_continuous_dynamics(self, x, u):
        """
        For dynamics xdot = f(x,u) = f0(x) + B(x)u returns:
        f = f(x,u): full dynamics
        A = df/dx(x,u): state Jacobian (note contains f0(x) and B(x) terms)
        B = B(x): control Jacobian
        """

        # f = jnp.zeros(self.grid.size) # Pre-allocate
        # f = f.at[1:].set(jnp.square(x))
        # f = f.at[0].set(f[-1])

        xdot_func = lambda x, u: -gudonov_flux(jnp.roll(x, 1), x, jnp.roll(x, -1)) / self.dx + \
                                 self.mu[0]*(jnp.roll(x, 1) - 2*x + jnp.roll(x, -1))/self.dx/self.dx + \
                                 jnp.dot(self.B, u)
        xdot = xdot_func(x, u)

        A, B = jax.jacobian(xdot_func, (0, 1))(x, u)

        return xdot, A, B

    # @partial(jax.jit, static_argnums=(0,))
    # def get_continuous_jacobians(self, x, u):
    #     return jax.jacobian(self.get_continuous_dynamics, (0, 1))(x, u)

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
        fd = viscous_burgers_implicit_step(self.grid, x, u, dt, self.mu)
        Ad, Bd = jax.jacobian(viscous_burgers_implicit_step, (1, 2))(self.grid, x, u, dt, self.mu)
        dd = fd - jnp.dot(Ad, x) - jnp.dot(Bd, u)

        return Ad, Bd, dd

    def get_characteristic_vals(self):
        """
        An optional function to define a procedure for computing characteristic values
        of the state and dynamics for use with GuSTO scaling, defaults to all ones
        """
        x_char = np.ones(self.n_x)
        f_char = np.ones(self.n_x)
        return x_char, f_char

    def get_next_state(self, x, u, dt):
        return viscous_burgers_implicit_step(self.grid, x, u, dt, self.mu, self.B)

    def rollout(self, x0, u, dt):
        """
        :x0: initial condition (n_x,)
        :u: array of control (N, n_u)
        :dt: time step

        Returns state x (N + 1, n_x) and performance variable z (N + 1, n_z)
        """
        N = u.shape[0]
        x = jnp.zeros((N + 1, self.n_x))

        # Set initial condition
        x.at[0, :].set(x0)

        # Simulate using some method
        for i in range(N):
            x.at[i + 1, :].set(np.reshape(self.get_next_state(x[i, :], u[i, :], dt), -1))
            print(" ... Working on timestep {}".format(i))

        # TODO: Currently full state observation
        z = x.copy()



        RuntimeError('Must be subclassed and implemented')
        return x, z

    def simulate(self, x0, controller, N, dt):
        ts = np.linspace(0, N*dt - dt, N)
        xs = np.zeros((N, self.n_x))
        zs = np.zeros((N, self.n_z))
        us = [None] * (N - 1)

        xs[0, :] = x0
        zs[0, :] = self.measurement_model.evaluate(x0)
        for j in range(N-1):
            print(" ... Solving timestep {}".format(j))
            x = xs[j, :]
            z = zs[j, :]
            t = ts[j]

            # TODO: Implement observer here
            controller.observer.update(us[j - 1], z, dt)

            x_belief = controller.observer.x
            topt, xopt, uopt, zopt, solve_time = controller.eval(x_belief, t)

            us[j] = uopt[0]

            # Simulate using built-in time-stepper
            xs[j + 1, :] = self.get_next_state(x, uopt[0], dt)
            zs[j + 1, :] = self.measurement_model.evaluate(xs[j + 1, :])

        return xs, us, ts

class Burgers_1D_ROM(Burgers_1D):
    """
        Burgers_1D model object for use with GuSTO class. This object describes continuous time dynamics:

        xdot = f(x,u) = f0(x) + B(x)u
        z = Hx
        """

    def __init__(self, n_x, n_u, n_z, mu, grid, V, observer):
        super(Burgers_1D_ROM, self).__init__(n_x, n_u, n_z, mu, grid, observer)
        self.V = V

        # Construct linear control matrix
        interval = int(n_x / n_u)
        eye = jnp.eye(n_u, n_u)
        self.B = jnp.kron(eye, jnp.ones(interval, )).T

    def get_continuous_dynamics(self, q, u):
        x = self.V @ q
        return super().get_continuous_dynamics(x, u)

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
        fd = viscous_burgers_implicit_step_rom(self.grid, x, u, dt, self.mu, self.B, self.V)
        Ad, Bd = jax.jacobian(viscous_burgers_implicit_step_rom, (1, 2))(self.grid, x, u, dt, self.mu, self.B, self.V)
        dd = fd - jnp.dot(Ad, x) - jnp.dot(Bd, u)

        return Ad, Bd, dd

    def get_next_state(self, x, u, dt):
        return viscous_burgers_implicit_step_rom(self.grid, x, u, dt, self.mu, self.B, self.V)

class Burgers_1D_HROM(Burgers_1D_ROM):
    """
        Burgers_1D model object for use with GuSTO class. This object describes continuous time dynamics:

        xdot = f(x,u) = f0(x) + B(x)u
        z = Hx
        """

    def __init__(self, n_x, n_u, n_z, mu, grid, V, weights, obsModel):
        super(Burgers_1D_ROM, self).__init__(n_x, n_u, n_z, mu, grid, obsModel)
        self.weights = weights
        self.V = V

        # Construct linear control matrix
        interval = int(V.shape[0] / n_u)
        eye = jnp.eye(n_u, n_u)
        B = jnp.kron(eye, jnp.ones(interval, )).T
        self.B = B

        self.mask = jnp.array(weights) > 0
        self.Vm = V[self.mask, :]
        self.Vl = V[jnp.roll(self.mask, 1), :]
        self.Vr = V[jnp.roll(self.mask, -1), :]
        # self.idxs = jnp.arange(weights.shape[0])[self.mask]
        self.H = obsModel.C @ self.V

    def get_continuous_dynamics(self, q, u):
        """
        For dynamics xdot = f(x,u) = f0(x) + B(x)u returns:
        f = f(x,u): full dynamics
        A = df/dx(x,u): state Jacobian (note contains f0(x) and B(x) terms)
        B = B(x): control Jacobian
        """

        mask = self.mask

        @jax.jit
        def qdot_func(q, u):
            w = self.Vm @ q
            w_l = self.Vl @ q
            w_r = self.Vr @ q
            return (-gudonov_flux(w_l, w, w_r) / self.dx + self.mu[0] * (
                    w_l - 2 * w + w_r) / self.dx / self.dx + jnp.dot(self.B[mask, :], u))

        qdot = qdot_func(q, u)

        A, B = jax.jacobian(qdot_func, (0, 1))(q, u)

        return A.T @ qdot, A.T @ A, A.T @ B

    def get_discrete_dynamics(self, q, u, dt):
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
        fd = viscous_burgers_implicit_step_hrom(self.grid, q, u, dt, self.mu, self.B, self.V, self.weights)
        Ad, Bd = jax.jacobian(viscous_burgers_implicit_step_hrom, (1, 2))(self.grid, q, u, dt, self.mu, self.B,
                                                                          self.V, self.weights)
        dd = fd - jnp.dot(Ad, q) - jnp.dot(Bd, u)

        return Ad, Bd, dd

    def get_next_state(self, q, u, dt):
        return viscous_burgers_implicit_step_hrom(self.grid, q, u, dt, self.mu, self.B, self.V, self.weights)
    
    def reduced_to_observed(self, x):
        return self.H @ x
    
    def xf_to_x(self, xf):
        return self.V.T @ xf

    def x_to_zfyf(self, x, zf=True):
        """
        :x: (N, n_x) or (n_x,) array
        :zf: boolean
        """
        return self.reduced_to_observed(x)

    def simulate(self, q0, controller, N, dt):
        ts = np.linspace(0, N*dt - dt, N)
        xs = np.zeros((N, self.V.shape[1]))
        us = [None] * (N - 1)

        xs[0, :] = q0
        for j in range(N-1):
            print(" ... Solving timestep {}".format(j))
            x = xs[j, :]
            t = ts[j]
            topt, xopt, uopt, zopt, solve_time = controller.eval(x, t)

            us[j] = uopt[0]

            # Simulate using built-in time-stepper
            xs[j + 1, :] = self.get_next_state(x, uopt[0], dt)

        return xs, us, ts
    
    def get_observer_jacobians(self,
                               x: jnp.ndarray):
        H = jax.jacobian(self.reduced_to_observed, 0)(x)
        c_res = self.reduced_to_observed(x) - jnp.dot(H, x)
        return H, c_res
    
    def update_dynamics(self, x, u, A_d, B_d, d_d):
        x_next = np.squeeze(A_d @ x) + np.squeeze(B_d @ u) + np.squeeze(d_d)
        return x_next