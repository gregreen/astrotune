"""Tests for the two solvers that fit the free harmonized parameters ``y``.

``gauss_newton`` (the default) solves the per-star normal equations with the
Jacobian transported through the ODE; ``gradient`` takes the step along the
loss's own gradient divided by the star's total precision. The Gauss-Newton step
is exact in one sweep, so the tests check the two properties that make it worth
the extra solve: it reaches the optimum, and it only touches its own stars.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sonnet")
pytest.importorskip("tensorflow_probability")
from astropy.table import Table  # noqa: E402

from astrotune import training_data as td  # noqa: E402


def _table(ids, d, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    return Table(
        {
            "id": np.asarray(ids, dtype="i8"),
            "x": rng.normal(size=(n, d)).astype("f4"),
            "sigma_x": np.full((n, d), 0.1, dtype="f4"),
            "x_err": np.full((n, d), 0.1, dtype="f4"),
        }
    )


def _small_trainer(**overrides):
    """A trainer small enough to build in a test, with every graph hook off.

    The second catalog repeats the first with a small offset, as real catalogs
    roughly do. Independent draws would put chi^2 far above ``chi2_turnover``,
    where the softening makes the step an IRLS iteration rather than a
    near-quadratic one, and the tests here are about the quadratic regime.
    """
    from astrotune.train_latent import LatentTrainConfig, LatentTrainer

    tables = [_table(np.arange(1200), 4, 0) for _ in range(2)]
    tables[1]["x"] = tables[0]["x"] + np.array([0.05, 0.1, 0.05, 0.05], dtype="f4")
    obs, ctrl = td.build_training_data(tables, seed=1)
    cfg = LatentTrainConfig(
        hidden_size=16, n_hidden=1, batch_size=256, n_epochs=1, max_steps=2,
        plot_every_seconds=None, plot_every_steps=None,
        checkpoint_every_seconds=None, agreement_n_stars=256, **overrides,
    )
    return LatentTrainer(obs, ctrl, cfg)


def _first_stars(tr, n=200):
    """The rows of the first few stars, which the file groups together."""
    return tr.x_obs[:n], tr.xs_obs[:n], tr.c_obs[:n], tr.star_obs[:n]


def _residuals(tr, y, x, xs, c):
    """Per-observation chi^2, the mapped value and its Jacobian at ``y``."""
    x_pred, _, jac = tr.bij.inverse_with_s_and_jac(y, c=c)
    chi2 = tf.reduce_sum((x - x_pred) ** 2 / xs**2, axis=1)
    return chi2, x_pred, jac


@pytest.mark.filterwarnings("ignore")
def test_gauss_newton_y_reaches_the_profile_optimum():
    """From a displaced start, one solve must land on the answer.

    ``initial_y`` already *is* the precision-weighted mean, which is the optimum
    for a map near the identity, so the start has to be displaced to test
    anything: otherwise the step is correctly zero.
    """
    tr = _small_trainer()
    x, xs, c, star = _first_stars(tr)

    uniq = np.unique(star.numpy())
    shift = tf.constant(np.full((len(uniq), 4), 0.15, dtype="f4"))
    tr.y.scatter_nd_add(tf.constant(uniq[:, None]), shift)

    chi2_before, _, _ = _residuals(tr, tf.gather(tr.y, star), x, xs, c)
    tr._gauss_newton_y(x, xs, c, star)
    chi2_after, x_pred1, jac1 = _residuals(tr, tf.gather(tr.y, star), x, xs, c)
    assert float(np.mean(chi2_after.numpy())) < 0.9 * float(np.mean(chi2_before.numpy()))

    # from the new point the step is a fixed point, so it must be far smaller
    # than the 0.15 displacement it has just undone
    A1, b1, _ = tr._normal_equations(x, xs, x_pred1, jac1, chi2_after, star)
    second = tr._solve_y(A1, b1).numpy()
    assert np.mean(np.abs(second)) < 0.05 * 0.15


@pytest.mark.filterwarnings("ignore")
def test_gauss_newton_y_touches_only_its_own_stars():
    tr = _small_trainer()
    x, xs, c, star = _first_stars(tr)
    y_before = tr.y.numpy().copy()

    tr._gauss_newton_y(x, xs, c, star)

    moved = np.flatnonzero(np.any(tr.y.numpy() != y_before, axis=1))
    assert set(moved.tolist()) == set(np.unique(star.numpy()).tolist())


@pytest.mark.filterwarnings("ignore")
def test_solve_y_ignores_a_direction_that_nothing_constrains():
    """A parameter that none of the star's catalogs measures leaves A singular.

    An unmeasured parameter contributes ``~1e-18`` to both ``A`` and ``b``, so
    the system is singular with a zero right-hand side: without the ridge the
    solve divides by zero and the *other* components come back as nan.
    """
    tr = _small_trainer()
    A = np.diag([2.0, 2.0, 2.0, 0.0])[None, :, :]
    b = np.array([[1.0, 1.0, 1.0, 0.0]])

    step = tr._solve_y(tf.constant(A), tf.constant(b)).numpy()[0]
    assert np.all(np.isfinite(step))
    assert np.allclose(step[:3], 0.5, atol=1e-5)  # the constrained directions
    assert abs(step[3]) < 1e-6  # the flat one stays put


@pytest.mark.filterwarnings("ignore")
def test_gradient_y_solver_still_trains(tmp_path):
    """The older solver stays available and still moves y."""
    tr = _small_trainer(y_solver="gradient")
    y0 = tr.y.numpy().copy()
    metrics = tr.train(tmp_path)
    assert len(metrics["loss"]) == 2
    assert np.all(np.isfinite(metrics["loss"]))
    assert not np.allclose(tr.y.numpy(), y0)


@pytest.mark.filterwarnings("ignore")
def test_gauss_newton_is_the_default_and_trains(tmp_path):
    """One graph-compiled run of the whole step, with no Jacobian complaints."""
    from astrotune.train_latent import LatentTrainConfig

    assert LatentTrainConfig().y_solver == "gauss_newton"
    tr = _small_trainer()
    y0 = tr.y.numpy().copy()
    metrics = tr.train(tmp_path)
    assert np.all(np.isfinite(metrics["loss"]))
    assert not np.allclose(tr.y.numpy(), y0)


def test_y_solver_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="y_solver"):
        _small_trainer(y_solver="newton")


def _displace_y(tr, amount=0.15):
    """Move every y off the precision-weighted mean, so a fit has work to do."""
    shift = tf.constant(np.full(tr.y.shape, amount, dtype="f4"))
    tr.y.assign_add(shift)
    return tr.y.numpy().copy()


@pytest.mark.filterwarnings("ignore")
def test_pretraining_leaves_y_at_its_initial_guess(tmp_path):
    """The pre-training phase must not move y at all, only the maps.

    ``initial_y`` is the precision-weighted mean, so holding it and fitting the
    bijections alone is the pre-training the maps are meant to start from.
    """
    tr = _small_trainer(pretrain_steps=10**6)  # longer than the whole run
    y0 = tr.y.numpy().copy()
    w0 = [v.numpy().copy() for v in tr.variables]

    metrics = tr.train(tmp_path)

    assert np.array_equal(tr.y.numpy(), y0)  # bit for bit, not approximately
    assert np.all(np.isfinite(metrics["loss"]))
    # and the map did move, so this is not simply a no-op
    assert any(not np.array_equal(v.numpy(), w) for v, w in zip(tr.variables, w0))


@pytest.mark.filterwarnings("ignore")
def test_y_moves_only_after_pretraining_ends(tmp_path):
    """With one pre-training step of two, the second step is the one that fits."""
    tr = _small_trainer(pretrain_steps=2)  # the whole 2-step run
    y_frozen = _displace_y(tr)
    tr.train(tmp_path)
    assert np.array_equal(tr.y.numpy(), y_frozen)

    tr = _small_trainer(pretrain_steps=1)  # pre-train one step, then fit
    y_mean = tr.y.numpy().copy()
    y_displaced = _displace_y(tr)
    tr.train(tmp_path)

    # some stars moved, and on average y is closer to the initial guess than the
    # displacement left it
    assert np.abs(tr.y.numpy() - y_displaced).max() > 0.0
    assert np.mean(np.abs(tr.y.numpy() - y_mean)) < np.mean(np.abs(y_displaced - y_mean))


@pytest.mark.filterwarnings("ignore")
def test_y_val_is_warm_started_between_validation_calls():
    """Each call must continue from the stored value, not from the initial guess.

    A half step makes the difference visible: on data this close to linear, one
    full sweep from either start lands in the same place, so a cold restart would
    be indistinguishable from a warm one.
    """
    tr = _small_trainer(y_learning_rate=0.5)
    x, xs, c, star = _first_stars(tr)
    y_train = tr.y.numpy().copy()

    # both arrays start at the same precision-weighted mean
    assert np.array_equal(tr.y_val.numpy(), y_train)
    uniq = np.unique(star.numpy())

    hist = []
    for _ in range(4):
        y_obs = tr._fit_y_val(x, xs, c, star)
        # the returned value is the state, and the state was written
        assert np.array_equal(y_obs.numpy(), tf.gather(tr.y_val, star).numpy())
        hist.append(tr.y_val.numpy()[uniq])

    gaps = [float(np.abs(b - a).max()) for a, b in zip(hist, hist[1:])]
    assert all(g > 0 for g in gaps), gaps  # every call moves it: not a cold restart
    assert gaps[0] > gaps[-1]  # and it is converging

    moved = np.flatnonzero(np.any(tr.y_val.numpy() != y_train, axis=1))
    assert set(moved.tolist()) == set(uniq.tolist())  # only this batch's stars
    assert np.array_equal(tr.y.numpy(), y_train)  # the training y is untouched


@pytest.mark.filterwarnings("ignore")
def test_y_val_is_saved_with_the_run(tmp_path):
    tr = _small_trainer()
    tr._fit_y_val(*_first_stars(tr))
    tr.save(tmp_path, {"loss": [1.0]})
    saved = np.load(tmp_path / "latent_y.npz")
    assert saved["y_val"].shape == saved["y"].shape
    assert not np.array_equal(saved["y_val"], saved["y"])  # they have diverged
