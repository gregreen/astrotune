"""Velocity-field network conditioned on a categorical variable."""

from __future__ import annotations

import logging

import numpy as np
import sonnet as snt
import tensorflow as tf

log = logging.getLogger(__name__)


class VectorNetwork(snt.Module):
    """Neural network mapping ``(x, c) -> y`` for a vector ``x`` and category ``c``.

    The network learns an embedding ``e`` for each category. Each hidden layer
    is modulated by ``e`` with FiLM (feature-wise linear modulation)::

        h = Dense1(x)
        h = Activation(h)
        h = LayerNorm(h)
        gamma, beta = split(Dense2(e))      # Dense2 output is 2x the size of h
        h = gamma * h + beta

    A final dense layer produces the output.
    """

    def __init__(
        self,
        n_cat,
        n_out,
        n_hidden,
        hidden_size,
        embed_dim=None,
        activation=tf.nn.silu,
        use_layer_norm=True,
        use_film=True,
        use_residual=True,
        output_scale=1e-5,
        output_bias=0,
        name=None,
    ):
        """
        Args:
          n_cat: number of categories.
          n_out: dimensionality of the output.
          n_hidden: number of hidden layers.
          hidden_size: number of neurons in each hidden layer.
          embed_dim: embedding size; defaults to ``ceil(log2(n_cat))``.
          activation: activation applied after each hidden layer.
          output_scale: stddev of the output-layer weight initialisation. A
            small value makes the initial velocity field ~0, i.e. the
            bijection starts out as the identity.
          output_bias: initial value of the output bias.
        """
        super().__init__(name=name)

        if embed_dim is None:
            embed_dim = int(np.ceil(np.log2(n_cat)))
            log.info("Using %d-dim embedding for %d categories.", embed_dim, n_cat)
        self.embed = snt.Embed(vocab_size=n_cat, embed_dim=embed_dim, name="embedding")

        self.hidden = [snt.Linear(hidden_size, name=f"hidden_{i}") for i in range(n_hidden)]
        if use_layer_norm:
            self.norm = [
                snt.LayerNorm(
                    axis=-1,
                    eps=1e-1,
                    create_scale=True,
                    create_offset=True,
                    name=f"norm_{i}",
                )
                for i in range(n_hidden)
            ]
        if use_film:
            self.film = [snt.Linear(2 * hidden_size, name=f"film_{i}") for i in range(n_hidden)]
            self.film_output = snt.Linear(
                n_out,
                w_init=snt.initializers.TruncatedNormal(stddev=output_scale),
                b_init=tf.zeros_initializer(),
                name="film_output",
            )

        self.output = snt.Linear(
            n_out,
            w_init=snt.initializers.TruncatedNormal(stddev=output_scale),
            b_init=tf.constant_initializer(value=output_bias),
            name="output",
        )

        self.n_cat = n_cat
        self.n_embed = embed_dim
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.use_film = use_film
        self.use_residual = use_residual

        self.n_calls = tf.Variable(0, dtype=tf.int32, trainable=False)

    def __call__(self, x, c):
        """Transform ``x, c -> y``.

        Args:
          x: float tensor with shape ``(b, n)``.
          c: integer tensor with shape ``(b,)``, in ``[0, n_cat-1]``.
        """
        self.n_calls.assign_add(1)

        e = self.embed(c)  # (b, embed_dim)

        for i, dense in enumerate(self.hidden):
            h = self.activation(dense(x))

            if self.use_layer_norm:
                h = self.norm[i](h)

            if self.use_film:
                gamma, beta = tf.split(self.film[i](e), 2, axis=-1)
                h = gamma * h + beta

            if self.use_residual and i > 0:
                x = x + h
            else:
                x = h

        y = self.output(x)

        if self.use_film:
            y = y + self.film_output(e)

        return y
