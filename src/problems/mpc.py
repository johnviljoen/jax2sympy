"""
Simplest constrained MPC problem I could think of
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jr

def linear():

    # Dynamics function
    nx, nu, N = 2, 1, 10
    z0 = jnp.ones(N*nx + (N-1)*nu)

    def dynamics(x, u):
        A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        B = jnp.array([[0.0], [1.0]])
        x_next = A @ x + B @ u
        return x_next

    # Objective function
    def f(z, nx=nx, nu=nu, N=N, Q=1.0, R=1.0):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        cost = jnp.sum(Q * x**2) + jnp.sum(R * (u**2))
        return cost

    # Equality constraints
    def h(z, nx=nx, nu=nu, N=N, x0=jnp.array([1., 1.])):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        constraints = []
        constraints.append(x[0] - x0)
        for k in range(N-1):
            x_next = dynamics(x[k], u[k])
            constraints.append(x[k+1] - x_next)
        return jnp.concatenate(constraints).flatten()

    # Inequality constraints
    def g(z, nx=nx, nu=nu, N=N):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        cl = []
        cl.append(x[:,0] - 0.5) # x >= 0.5
        return jnp.hstack(cl)

    #
    x0 = z0

    gt = None

    return f, h, g, x0, gt, []

def nonlinear():

    nx, nu, N = 4, 2, 20

    # Dynamics function
    def dynamics(x, u):
        # {x, xd, y, yd}, {ux, uy}
        A = jnp.array([
            [1.0, 1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        B = jnp.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ])
        x_next = A @ x + B @ u
        return x_next

    # Objective function
    def f(z, nx=nx, nu=nu, N=N, Q=1.0, R=10.0):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        cost = jnp.sum(Q * x**2) + jnp.sum(R * (u**2))
        return cost

    # Equality constraints
    def h(z, nx=nx, nu=nu, N=N, x0=jnp.array([2., 0.0, 2., 0.0])):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        constraints = []
        constraints.append(x[0] - x0)
        for k in range(N-1):
            x_next = dynamics(x[k], u[k])
            constraints.append(x[k+1] - x_next)
        # assign final x to be at the end
        # constraints.append(x[k+1] - jnp.zeros([4]))
        return jnp.concatenate(constraints).flatten()

    # Inequality constraints
    def g(z, nx=nx, nu=nu, N=N):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        cl = []
        # cl.append(u[:,0] + 0.1)
        # cl.append(-u[:,0] + 0.1)
        # cl.append(u[:,1] + 0.1)
        # cl.append(-u[:,1] + 0.1)
        cl.append(x[:,1] + 0.2)
        cl.append(x[:,3] + 0.2)
        cl.append(-x[:,1] + 0.2)
        cl.append(-x[:,3] + 0.2)
        for k in range(N-1):
            multiplier = 1 + k * 0.1
            x_next = dynamics(x[k], u[k])
            cl.append(jnp.sqrt((x_next[0] - 1.0)**2 + (x_next[2] - 1.0)**2) - 0.5 * multiplier)
        return jnp.hstack(cl)

    def xu_to_z(x, u):
        return jnp.hstack([x.flatten(), u.flatten()])

    def z_to_xu(z):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        return x, u

    # x0 = jnp.ones(N*nx + (N-1)*nu) * 2
    x0 = xu_to_z(jnp.array([
        [2,0,2.01,0] * N
    ]), jnp.array([
        [0.,0.] * (N-1)
    ]))

    gt = None

    aux = [z_to_xu, xu_to_z]

    return f, h, g, x0, gt, aux

def quadcopter_nav(N=3):

    # Dynamics setup
    nx, nu, Ts = 13, 4, 0.1

    # fundamental quad parameters
    qp = {
        "mB": 1.2, # mass (kg)
        "dxm": 0.16, # arm length (m)
        "dym": 0.16, # arm length (m)
        "dzm": 0.01, # arm height (m)
        "IB": np.array([[0.0123, 0,      0     ],
                        [0,      0.0123, 0     ],
                        [0,      0,      0.0224]]), # Inertial tensor (kg*m^2)
        "IRzz": 2.7e-5, # rotor moment of inertia (kg*m^2)
        "Cd": 0.1, # drag coefficient (omnidirectional)
        "kTh": 1.076e-5, # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        "kTo": 1.632e-7, # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
        "minThr": 0.1*4, # Minimum total thrust (N)
        "maxThr": 9.18*4, # Maximum total thrust (N)
        "minWmotor": 75, # Minimum motor rotation speed (rad/s)
        "maxWmotor": 925, # Maximum motor rotation speed (rad/s)
        "tau": 0.015, # Value for second order system for Motor dynamics
        "kp": 1.0, # Value for second order system for Motor dynamics
        "damp": 1.0, # Value for second order system for Motor dynamics
        "usePrecession": True, # model precession or not
        # "w_hover": 522.9847140714692, # hardcoded hover rotor speed (rad/s)
    }

    # post init useful parameters for quad
    qp["B0"] = np.array([
        [qp["kTh"], qp["kTh"], qp["kTh"], qp["kTh"]],
        [qp["dym"]*qp["kTh"], -qp["dym"]*qp["kTh"], -qp["dym"]*qp["kTh"], qp["dym"]*qp["kTh"]],
        [qp["dxm"]*qp["kTh"], qp["dxm"]*qp["kTh"], -qp["dxm"]*qp["kTh"], -qp["dxm"]*qp["kTh"]],
        [-qp["kTo"], qp["kTo"], -qp["kTo"], qp["kTo"]]]) # actuation matrix

    qp["x_lb"] = np.array([
        *[-10]*3, *[-np.inf]*4, *[-10]*3, *[-10]*3, *[qp["minWmotor"]]*4
        # xyz       q0123         xdydzd    pdqdrd    w0123
    ])

    qp["x_ub"] = np.array([
        *[10]*3, *[np.inf]*4, *[10]*3, *[10]*3, *[qp["maxWmotor"]]*4
        # xyz      q0123        xdydzd   pdqdrd   w0123
    ])

    qp["K_p_w"] = 0.001 / Ts * 0.05 # Rotor proportional gain

    def quad_f_step(x, u, d=jnp.zeros(3)):

        # u is the demanded rotational rates of the rotors

        xd = quad_f(x, u, d, qp)
        x_next = x + xd * Ts

        # normalize quaternion
        x_next = x_next.at[3:7].set(x_next[3:7] / jnp.linalg.norm(x_next[3:7]))
        # clip rotor speed
        # x_next = x_next.at[13:17].set(jnp.clip(x_next[13:17], qp["x_lb"][13:17], qp["x_ub"][13:17]))

        return x_next

    def quad_f(x, u, d, qp, g=9.81):

        # instantaneous thrusts and torques generated by the current w0...w3
        # x[13:17] = jnp.clip(x[13:17], qp["minWmotor"], qp["maxWmotor"]) # this clip shouldn't occur within the dynamics
        th = qp['kTh'] * u ** 2
        to = qp['kTo'] * u ** 2

        # state derivates (from sympy.mechanics derivation)
        # -------------------------------------------------
        xd = jnp.stack(
            [
                x[7], x[8], x[9], # xd, yd, zd
                - 0.5 * x[10] * x[4] - 0.5 * x[11] * x[5] - 0.5 * x[6] * x[12], # q0d
                0.5 * x[10] * x[3] - 0.5 * x[11] * x[6] + 0.5 * x[5] * x[12], # q1d
                0.5 * x[10] * x[6] + 0.5 * x[11] * x[3] - 0.5 * x[4] * x[12], # q2d
                - 0.5 * x[10] * x[5] + 0.5 * x[11] * x[4] + 0.5 * x[3] * x[12], # q3d
                - (
                    qp["Cd"]
                    * jnp.sign(d[0] * jnp.cos(d[1]) * jnp.cos(d[2]) - x[7])
                    * (d[0] * jnp.cos(d[1]) * jnp.cos(d[2]) - x[7]) ** 2
                    - 2 * (x[3] * x[5] + x[4] * x[6]) * (th[0] + th[1] + th[2] + th[3])
                )
                / qp["mB"], # xdd
                - (
                    qp["Cd"]
                    * jnp.sign(d[0] * jnp.sin(d[1]) * jnp.cos(d[2]) - x[8])
                    * (d[0] * jnp.sin(d[1]) * jnp.cos(d[2]) - x[8]) ** 2
                    + 2 * (x[3] * x[4] - x[5] * x[6]) * (th[0] + th[1] + th[2] + th[3])
                )
                / qp["mB"], # ydd
                - (
                    -qp["Cd"] * jnp.sign(d[0] * jnp.sin(d[2]) + x[9]) * (d[0] * jnp.sin(d[2]) + x[9]) ** 2
                    - (th[0] + th[1] + th[2] + th[3])
                    * (x[3] ** 2 - x[4] ** 2 - x[5] ** 2 + x[6] ** 2)
                    + g * qp["mB"]
                )
                / qp["mB"], # zdd (the - in front turns increased height to be positive - SWU)
                (
                    (qp["IB"][1,1] - qp["IB"][2,2]) * x[11] * x[12]
                    - qp["usePrecession"] * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3]) * x[11]
                    + (th[0] - th[1] - th[2] + th[3]) * qp["dym"]
                )
                / qp["IB"][0,0], # pd
                (
                    (qp["IB"][2,2] - qp["IB"][0,0]) * x[10] * x[12]
                    + qp["usePrecession"] * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3]) * x[10]
                    + (th[0] + th[1] - th[2] - th[3]) * qp["dxm"]
                )
                / qp["IB"][1,1], #qd
                ((qp["IB"][0,0] - qp["IB"][1,1]) * x[10] * x[11] - to[0] + to[1] - to[2] + to[3]) / qp["IB"][2,2], # rd
                # tau[0] / qp["IRzz"], tau[1] / qp["IRzz"], tau[2] / qp["IRzz"], tau[3] / qp["IRzz"] # w0d ... w3d
            ]
        )

        return xd

    def xu_to_z(x, u):
        return jnp.hstack([x.flatten(), u.flatten()])

    def z_to_xu(z):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        return x, u

    state0 = jnp.array([[2,2,0,1,0,0,0,0,0,0,0,0,0]]*N)
    input0 = jnp.array([[522.9847140714692]*4]*(N-1))
    z_init = xu_to_z(state0, input0)

    # Objective function
    def f(z, Q=1.0, R=0.0):
        x, u = z_to_xu(z)
        cost = jnp.sum(Q * x[:,:3]**2) + jnp.sum(Q * x[:,7:]**2) + jnp.sum(R * (u**2))
        return cost

    # Equality constraints
    def h(z, N=N, x0=state0[0]):
        x, u = z_to_xu(z)
        constraints = []
        constraints.append(x[0] - x0)
        for k in range(N-1):
            x_next = quad_f_step(x[k], u[k])
            constraints.append(x[k+1] - x_next)
        return jnp.concatenate(constraints).flatten()

    # Inequality constraints
    def g(z, N=N):
        x, u = z_to_xu(z)
        cl = []
        # cl.append(x[:,0] - 0.5) # x >= 0.5
        for k in range(N-1):
            multiplier = 1 + k * 0.1
            x_next = quad_f_step(x[k], u[k])
            # cl.append(jnp.sqrt((x_next[0] - 1.0)**2 + (x_next[1] - 1.0)**2) - 0.5 * multiplier)
            cl.append(u[k] - qp["minWmotor"]) # u > minW
            cl.append(qp["maxWmotor"] - u[k]) # maxW > u
        return jnp.hstack(cl)

    gt = None

    aux = {
        "z_to_xu": z_to_xu, 
        "xu_to_z": xu_to_z, 
        "qp": qp, 
        "N": N, 
        "nx": nx, 
        "nu": nu,
        "quad_f_step": quad_f_step
    }

    return f, h, g, z_init, gt, aux

if __name__ == "__main__":

    from problems.quad_plot import Animator
    from jaxipm.ipopt_reference import run_ipopt

    f, h, g, z_init, gt, aux = quadcopter_nav()

    result = run_ipopt(f, h, g, z_init)

    x = aux[0](result.x)[0]
    p = aux[2]
    N = 10
    Ts = 0.1
    t = np.arange(0, N * Ts, Ts)
    r = np.array([[0,0,0,1,0,0,0,0,0,0,0,0,0]]*N)

    animator = Animator(p, x, t, r)
    animator.animate()

    print('fin')