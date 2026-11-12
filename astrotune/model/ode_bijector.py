"""Bijector defined by integrating a catalog-conditioned velocity field."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.internal import prefer_static

from .vector_network import VectorNetwork

tfb = tfp.bijectors


class ODEBijector(tfb.Bijector):
    r"""Bijection :math:`x \mapsto y = x(1)` with :math:`dx/dt = v_c(t, x)`, :math:`x(0) = x`.

    The inverse integrates the same ODE backwards from ``t = 1`` to ``t = 0``.
    Besides the plain maps (``_forward``/``_inverse``) there are variants that
    also return the integrated path length ``s`` (used as a training penalty),
    the same with the Jacobian ``d x(1)/d x(0)`` (``forward_with_s_and_jac`` and
    ``inverse_with_s_and_jac``), and ``augmented_forward``, which adds
    ``ln det J``.

    The Jacobian is *transported* as part of the ODE state rather than obtained
    by differentiating the solver: with ``M(t) = d x(t)/d x(0)``,

    .. math:: \\dot M = \\frac{\\partial v}{\\partial x}\\, M,\\qquad M(0) = I,

    so it comes out of the same integration for the price of one extra matrix
    multiply per step. Nothing differentiated ``M`` afterwards: it is only used
    to take a Gauss-Newton step in the harmonized parameters, so it is never
    handed to a ``gradient`` call and needs no ``stop_gradient``.
    """

    def __init__(
        self,
        n_dim,
        n_cond,
        hidden_size,
        n_hidden,
        rtol=1e-5,
        atol=1e-5,
        vector_network_kwargs=None,
        validate_args=False,
        name="ode_bijector",
    ):
        super().__init__(validate_args=validate_args, forward_min_event_ndims=0, name=name)

        self.n_dim = n_dim
        self.v = VectorNetwork(
            n_cond, n_dim, n_hidden, hidden_size, **(vector_network_kwargs or {})
        )
        self.v(tf.zeros([1, 1 + n_dim]), tf.zeros([1], dtype=tf.int32))  # create variables

        self.ode_solver = tfp.math.ode.DormandPrince(rtol=rtol, atol=atol)

    def conditional_velocity(self, t, x, c):
        """Velocity at time ``t`` and position ``x`` for category ``c``."""
        # Shape (b, 1), even when x.shape = (None, n)
        shape = tf.where([True, False], tf.shape(x), [0, 1])
        tx = tf.concat([tf.broadcast_to(t, shape), x], 1)
        return self.v(tx, c)

    def augmented_ode(self, t, xsJ, c):
        x, s, lndetJ = tf.split(xsJ, [-1, 1, 1], axis=1)

        dx_dt, diag_J = tfp_math.diag_jacobian(
            xs=x,
            fn=lambda xx: self.conditional_velocity(t, xx, c),
            sample_shape=[prefer_static.size0(x)],
        )
        if isinstance(dx_dt, list):
            dx_dt = dx_dt[0]
        if isinstance(diag_J, list):
            diag_J = diag_J[0]

        # d(lndetJ)/dt = tr(J); the factor 2 follows the reference notebook
        dlndetJ_dt = 2 * tf.math.reduce_sum(diag_J, axis=1, keepdims=True)
        ds_dt = tf.math.sqrt(tf.reduce_sum(dx_dt**2, axis=1, keepdims=True))

        return tf.concat([dx_dt, ds_dt, dlndetJ_dt], 1)

    def ode_with_s(self, t, xs, c):
        x, _ = tf.split(xs, [-1, 1], axis=1)
        dx_dt = self.conditional_velocity(t, x, c)
        ds_dt = tf.math.sqrt(tf.reduce_sum(dx_dt**2, axis=1, keepdims=True))
        return tf.concat([dx_dt, ds_dt], 1)

    def velocity_and_jacobian(self, t, x, c):
        """The velocity and its Jacobian with respect to ``x``.

        Returns:
          v: ``(b, n)`` velocity at ``(t, x)``, as ``conditional_velocity``.
          jac: ``(b, n, n)`` with ``jac[b, i, j] = d v_i / d x_j``.
        """
        # One forward pass and one backward pass per component. ``unstack`` has
        # to happen inside the tape: a slice taken afterwards is not a recorded
        # operation, so ``gradient`` would quietly return None for it.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            v = self.conditional_velocity(t, x, c)
            components = tf.unstack(v, axis=1)
        rows = [tape.gradient(component, x) for component in components]
        del tape  # every row is extracted; release what the tape holds
        zeros = tf.zeros_like(x)
        return v, tf.stack([zeros if row is None else row for row in rows], axis=1)

    def ode_with_s_and_jac(self, t, state, c, sign):
        """Right-hand side of the state ``[x, s, M]``, ``M = d x(t)/d x(0)``.

        ``sign`` is ``+1`` for the forward map (which integrates
        ``v(t, x)``) and ``-1`` for the inverse (which integrates
        ``-v(1 - t, x)``).
        """
        n = self.n_dim
        x = state[:, :n]
        v, jac = self.velocity_and_jacobian(t if sign > 0 else 1 - t, x, c)

        ds_dt = tf.math.sqrt(tf.reduce_sum(v**2, axis=1, keepdims=True))
        M = tf.reshape(state[:, n + 1 :], [-1, n, n])
        dM_dt = tf.linalg.matmul(jac, M)

        return tf.concat(
            [sign * v, sign * ds_dt, sign * tf.reshape(dM_dt, [-1, n * n])], 1
        )

    def integrate_with_jac(self, x0, c, sign):
        """Integrate ``[x, s, M]`` from ``x(0) = x0`` to ``t = 1``."""
        n = self.n_dim
        b = tf.shape(x0)[0]
        identity = tf.reshape(tf.eye(n, dtype=x0.dtype), [1, n * n])
        state0 = tf.concat(
            [
                x0,
                tf.zeros([b, 1], dtype=x0.dtype),
                tf.broadcast_to(identity, tf.stack([b, n * n])),
            ],
            1,
        )
        dxsM_dt = lambda t, state: self.ode_with_s_and_jac(t, state, c, sign)
        res = self.ode_solver.solve(dxsM_dt, 0.0, state0, [1.0])
        state = tf.squeeze(res.states, 0)
        return (
            state[:, :n],
            state[:, n : n + 1],
            tf.reshape(state[:, n + 1 :], [-1, n, n]),
        )

    def _forward(self, x, c=None):
        dx_dt = lambda tt, xx: self.conditional_velocity(tt, xx, c)
        res = self.ode_solver.solve(dx_dt, 0.0, x, [1.0])
        return tf.squeeze(res.states, 0)

    def _inverse(self, y, c=None):
        dx_dt = lambda tt, xx: -self.conditional_velocity(1 - tt, xx, c)
        res = self.ode_solver.solve(dx_dt, 0.0, y, [1.0])
        return tf.squeeze(res.states, 0)

    def forward_with_s(self, x, c=None):
        """Map ``x`` to the harmonized space; also return the path length."""
        dxs_dt = lambda t, xs: self.ode_with_s(t, xs, c)
        xs0 = tf.concat([x, tf.zeros([tf.shape(x)[0], 1])], 1)
        res = self.ode_solver.solve(dxs_dt, 0.0, xs0, [1.0])
        ys = tf.squeeze(res.states, 0)
        y, s = tf.split(ys, [-1, 1], axis=1)
        return y, s

    def inverse_with_s(self, y, c=None):
        """Map ``y`` from the harmonized space back; also return the path length."""
        dxs_dt = lambda t, xs: -self.ode_with_s(1 - t, xs, c)
        xs0 = tf.concat([y, tf.zeros([tf.shape(y)[0], 1])], 1)
        res = self.ode_solver.solve(dxs_dt, 0.0, xs0, [1.0])
        xs = tf.squeeze(res.states, 0)
        x, s = tf.split(xs, [-1, 1], axis=1)
        return x, -s

    def forward_with_s_and_jac(self, x, c=None):
        """Forward map; also the path length and ``d y / d x``.

        Returns:
          y: ``(b, n)`` harmonized parameters.
          s: ``(b, 1)`` path length.
          jac: ``(b, n, n)`` with ``jac[b, i, j] = d y_i / d x_j``.
        """
        return self.integrate_with_jac(x, c, 1.0)

    def inverse_with_s_and_jac(self, y, c=None):
        """Inverse map; also the path length and ``d x / d y``.

        Returns:
          x: ``(b, n)`` catalog-space parameters.
          s: ``(b, 1)`` path length.
          jac: ``(b, n, n)`` with ``jac[b, i, j] = d x_i / d y_j``.
        """
        x, s, jac = self.integrate_with_jac(y, c, -1.0)
        return x, -s, jac

    def augmented_forward(self, x, c=None):
        """Forward map that also returns the path length and ``ln det J``."""
        dxsJ_dt = lambda t, xsJ: self.augmented_ode(t, xsJ, c)
        n = tf.shape(x)[0]
        xsJ0 = tf.concat([x, tf.zeros([n, 1]), tf.zeros([n, 1])], 1)
        res = self.ode_solver.solve(dxsJ_dt, 0.0, xsJ0, [1.0])
        ysJ = tf.squeeze(res.states, 0)
        y, s, lndetJ = tf.split(ysJ, [-1, 1, 1], axis=1)
        return y, s, lndetJ

    def return_path(self, x, c, n_timesteps):
        """Trajectory of ``x`` at ``n_timesteps`` evenly spaced times in [0, 1]."""
        t_solution = tf.linspace(0.0, 1.0, n_timesteps)
        dx_dt = lambda tt, xx: self.conditional_velocity(tt, xx, c)
        res = self.ode_solver.solve(dx_dt, 0.0, x, t_solution)
        return res.states

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        # Use ``augmented_forward`` to obtain ln det J.
        raise NotImplementedError()
