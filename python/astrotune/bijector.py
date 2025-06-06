
class CatMLP(snt.Module):
    """
    Categorical multi-layer perceptron. This is very similar to a standard
    feed-forward neural network, except that it also takes an integer-valued
    categorical variable, c, as input. The transformation looks like:

      CatMLP: x, c -> y

    where x in R^n, c in Z, and y in R^m.
    """
    
    def __init__(self, n_cat, n_out, n_hidden, hidden_size,
                 activation=tf.math.tanh, name=None):
        """
        Constructor for CatMLP.

        Inputs:
          n_cat (int): # of categories.
          n_out (int): Dimensionality of output (m).
          n_hidden (int): # of hidden layers.
          hidden_size (int): # of neurons in each hidden layer.
          activation (Optional[function]): Activation function to apply after
            each hidden layer.
          name (Optional[str]): Name to give to model.
        """
        super(CatMLP, self).__init__(name=name)
        self.hidden = [
            snt.Linear(hidden_size, name=f'hidden{i}')
            for i in range(n_hidden)
        ]
        self.output = snt.Linear(n_out, name='output')
        self.n_cat = n_cat
        self.activation = activation

    def __call__(self, x, c):
        """
        Transform x,c -> y.

        Inputs:
          x (tf.Tensor): Float-valued tensor with shape (b,n), where n is the
            dimensionality of the input and b is the batch dimension.
          c (tf.Tensor): Integer-valued tensor with shape (b,), where b is
            the batch dimension. This is the categorical variable. It should
            be in the range [0, n_cat-1].
        """
        c = tf.one_hot(c, self.n_cat)
        y = tf.concat([x, c], axis=1)
        for i,h in enumerate(self.hidden):
            if i != 0:
                y = tf.concat([y,x,c], axis=1)
            y = h(y)
            y = self.activation(y)
        y = self.output(y)
        return y


class ODEBijector(tfb.Bijector):
    def __init__(self, n_dim, n_cond, hidden_size, n_hidden,
                 rtol=1e-5, atol=1e-5,
                 validate_args=False, name='ode_bijector'):
        super(ODEBijector, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=0,
            name=name
        )

        # Velocity field
        self.v = CatMLP(n_cond, n_dim, n_hidden, hidden_size)

        # Initialize the velocity field
        self.v(tf.zeros([1,1+n_dim]), tf.zeros([1], dtype=tf.int32))

        # ODE solver
        self.ode_solver = tfp.math.ode.DormandPrince(rtol=rtol, atol=atol)

    def conditional_velocity(self, t, x, c):
        # The following returns (x.shape[0], 1), even when x.shape = (None,n)
        shape = tf.where([True, False], tf.shape(x), [0, 1])
        # Concatenate t and x
        tx = tf.concat([tf.broadcast_to(t,shape), x], 1)
        # Return the velocity at time t and position x, given category c
        return self.v(tx, c)
    
    def augmented_ode(self, t, xsJ, c):
        x,s,lndetJ = tf.split(xsJ, [-1,1,1], axis=1)

        # dx_dt = self.conditional_velocity(t, x, c)
        dx_dt, diag_J = tfp_math.diag_jacobian(
            xs=x,
            fn=lambda xx: self.conditional_velocity(t, xx, c),
            sample_shape=[prefer_static.size0(x)]
        )
        if isinstance(dx_dt, list):
            dx_dt = dx_dt[0]
        if isinstance(diag_J, list):
            diag_J = diag_J[0]

        # d(lndetJ)/dt = tr(J)
        dlndetJ_dt = 2*tf.math.reduce_sum(diag_J, axis=1, keepdims=True)
        
        # ds/dt = |dx/dt|
        ds_dt = tf.math.sqrt(tf.reduce_sum(dx_dt**2, axis=1, keepdims=True))

        # Combine derivatives into one vector
        dxsJ_dt = tf.concat([dx_dt, ds_dt, dlndetJ_dt], 1)

        return dxsJ_dt
    
    def ode_with_s(self, t, xs, c):
        x,s = tf.split(xs, [-1,1], axis=1)

        # dx_dt = self.conditional_velocity(t, x, c)
        dx_dt = self.conditional_velocity(t, x, c)
        
        # ds/dt = |dx/dt|
        ds_dt = tf.math.sqrt(tf.reduce_sum(dx_dt**2, axis=1, keepdims=True))

        # Combine derivatives into one vector
        dxs_dt = tf.concat([dx_dt, ds_dt], 1)

        return dxs_dt
    
    def _forward(self, x, c=None):
        dx_dt = lambda tt, xx: self.conditional_velocity(tt, xx, c)
        res = self.ode_solver.solve(dx_dt, 0., x, [1.])
        y = tf.squeeze(res.states, 0)
        return y

    def _inverse(self, y, c=None):
        dx_dt = lambda tt, xx: -self.conditional_velocity(1-tt, xx, c)
        res = self.ode_solver.solve(dx_dt, 0., y, [1.])
        x = tf.squeeze(res.states, 0)
        return x
    
    def forward_with_s(self, x, c=None):
        dxs_dt = lambda t, xs: self.ode_with_s(t, xs, c)
        xs0 = tf.concat([
            x,                        # x_0 = x
            tf.zeros([x.shape[0],1]), # s_0 = 0
        ], 1)
        res = self.ode_solver.solve(dxs_dt, 0., xs0, [1.])
        ys = tf.squeeze(res.states, 0)
        y,s = tf.split(ys, [-1,1], axis=1)
        return y, s
    
    def inverse_with_s(self, y, c=None):
        dxs_dt = lambda t, xs: -self.ode_with_s(1-t, xs, c)
        xs0 = tf.concat([
            y,                        # x_0 = y
            tf.zeros([y.shape[0],1]), # s_0 = 0
        ], 1)
        res = self.ode_solver.solve(dxs_dt, 0., xs0, [1.])
        xs = tf.squeeze(res.states, 0)
        x,s = tf.split(xs, [-1,1], axis=1)
        return x, -s
    
    def augmented_forward(self, x, c=None):
        dxsJ_dt = lambda t, xsJ: self.augmented_ode(t, xsJ, c)
        xsJ0 = tf.concat([
            x,                        # x_0 = x
            tf.zeros([x.shape[0],1]), # s_0 = 0
            tf.zeros([x.shape[0],1])  # lndetJ_0 = 0
        ], 1)
        res = self.ode_solver.solve(dxsJ_dt, 0., xsJ0, [1.])
        ysJ = tf.squeeze(res.states, 0)
        y,s,lndetJ = tf.split(ysJ, [-1,1,1], axis=1)
        return y, s, lndetJ

    def return_path(self, x, c, n_timesteps):
        t_solution = tf.linspace(0., 1., n_timesteps)
        dx_dt = lambda tt, xx: self.conditional_velocity(tt, xx, c)
        res = self.ode_solver.solve(dx_dt, 0., x, t_solution)
        y_t = res.states
        return y_t

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        # Notice that we needn't do any reducing, even when`event_ndims > 0`.
        # The base Bijector class will handle reducing for us; it knows how
        # to do so because we called `super` `__init__` with
        # `forward_min_event_ndims = 0`.
        raise NotImplementedError()
