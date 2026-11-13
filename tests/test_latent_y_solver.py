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
    settings = dict(
        hidden_size=16, n_hidden=1, batch_size=256, n_epochs=1, max_steps=2,
        plot_every_seconds=None, plot_every_steps=None,
        checkpoint_every_steps=None, agreement_n_stars=256,
    )
    settings.update(overrides)
    return LatentTrainer(obs, ctrl, LatentTrainConfig(**settings))


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
    # the constrained directions, damped by the ridge (y_ridge = 1e-3 times the
    # mean diagonal, 1.5): 2 / (2 + 1.5e-3)
    assert np.allclose(step[:3], 0.5, atol=1e-3)
    assert abs(step[3]) < 1e-6  # the flat one stays put


@pytest.mark.filterwarnings("ignore")
def test_trust_region_caps_a_runaway_step_and_keeps_its_direction():
    """A step that would move a star by a whole unit is scaled, not clipped.

    The direction has to survive: each component is cut by the same factor, so
    the scaled step is still the exact minimiser along that line. Clipping per
    dimension would bend the step and could even raise the chi^2.
    """
    tr = _small_trainer()
    # curvature 1e-4 in the last direction against 2 in the others: the solve
    # puts a step of ~1e4 there, i.e. a star flung across the parameter space
    A = np.diag([2.0, 2.0, 2.0, 1e-4])[None, :, :]
    b = np.array([[1.0, 1.0, 1.0, 1.0]])

    raw, allowed = tr._solve_y(tf.constant(A), tf.constant(b), return_scale=True)
    raw, allowed = raw.numpy()[0], float(allowed.numpy()[0])
    assert np.abs(raw).max() <= tr.cfg.y_trust_abs + 1e-6
    assert allowed < 1e-3  # the cap is what did it

    # the direction is the one the unconstrained solve gave: every component was
    # cut by the same factor
    free = _small_trainer(y_trust_abs=None)._solve_y(A, b)
    ratio = raw / free.numpy()[0]
    assert np.allclose(ratio, ratio[0], rtol=1e-4), ratio


def test_trust_region_leaves_an_ordinary_step_alone():
    tr = _small_trainer()
    x, xs, c, star = _first_stars(tr)
    x_pred, _, jac = tr.bij.inverse_with_s_and_jac(tf.gather(tr.y, star), c=c)
    chi2 = tf.reduce_sum((x - x_pred) ** 2 / xs**2, axis=1)
    A, b, _ = tr._normal_equations(x, xs, x_pred, jac, chi2, star)

    blocked = _small_trainer(y_trust_abs=None)
    step, allowed = tr._solve_y(A, b, return_scale=True)
    free = blocked._solve_y(A, b)
    assert float(np.min(allowed.numpy())) == 1.0
    assert np.allclose(step.numpy(), free.numpy())


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


@pytest.mark.filterwarnings("ignore")
def test_validation_batches_are_a_random_sample_not_an_id_block():
    """A validation batch must be a random sample of the sky, as the training
    batches are. Row indices follow the sorted Gaia ids, so an unshuffled batch is
    a contiguous id block -- a patch of sky -- and its chi^2 then varies several
    times more from batch to batch than the training chi^2 does."""
    tr = _small_trainer()
    obs_idx, _ = next(iter(tr.val_data))
    idx = obs_idx.numpy()
    assert len(idx) > 10
    assert not np.all(np.diff(idx) > 0), "validation rows are still in id order"


@pytest.mark.filterwarnings("ignore")
def test_lr_warms_up_before_it_decays():
    """The rate must reach the peak only at the end of the warmup, and decay after.

    Adam's step is ``~lr`` whatever the gradient norm is, so the first segment of
    the schedule is the one that decides how far from its start the run can
    travel: from the first step to the first drop, everything done at the full
    rate. The ramp makes the early steps progressively less violent.
    """
    from astrotune.train_latent import _warmup_and_decay

    peak, warmup = 1e-3, 50
    decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [500], [peak, peak / 2]
    )
    sched = _warmup_and_decay(decay, peak, warmup)
    at = lambda s: float(sched(tf.constant(s, tf.int64)))  # noqa: E731

    assert at(0) == pytest.approx(peak / warmup)  # the very first step
    assert at(24) == pytest.approx(peak * 25 / warmup)
    assert at(49) == pytest.approx(peak)  # the ladder reaches the peak
    assert at(300) == pytest.approx(peak)  # and holds it
    assert at(600) == pytest.approx(peak / 2)  # until the decay takes over

    ramp = [at(s) for s in range(warmup)]
    assert all(b >= a for a, b in zip(ramp, ramp[1:])), "the ramp must not wobble"


@pytest.mark.filterwarnings("ignore")
def test_the_warmup_schedule_is_one_adam_accepts():
    """Keras passes the step only to a ``LearningRateSchedule``.

    A plain callable is called with no arguments (``base_optimizer.
    _get_current_learning_rate``), which is a ``TypeError`` at the first
    ``apply_gradients`` and not before -- the run has already loaded 1.4M star
    vectors by then. So this is worth a step of actual Adam.
    """
    from keras.optimizers.schedules import LearningRateSchedule
    from astrotune.train_latent import _warmup_and_decay

    decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [500], [1e-3, 5e-4]
    )
    sched = _warmup_and_decay(decay, 1e-3, 50)
    assert isinstance(sched, LearningRateSchedule)

    v = tf.Variable([1.0, 2.0])
    opt = tf.keras.optimizers.Adam(learning_rate=sched)
    opt.build([v])
    for _ in range(3):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(v**2)
        opt.apply_gradients([(tape.gradient(loss, [v])[0], v)])
    assert int(opt.iterations) == 3
    assert np.all(np.abs(v.numpy()) < np.array([1.0, 2.0]))  # it did move


@pytest.mark.filterwarnings("ignore")
def test_zero_warmup_is_the_plain_decay():
    from astrotune.train_latent import _warmup_and_decay

    decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [500], [1e-3, 5e-4]
    )
    sched = _warmup_and_decay(decay, 1e-3, 0)
    assert sched is decay
    for step in (0, 499, 500):
        assert float(sched(tf.constant(step, tf.int64))) == float(
            decay(tf.constant(step, tf.int64))
        )


@pytest.mark.filterwarnings("ignore")
def test_the_trainer_reports_how_far_the_trust_region_bound(tmp_path):
    """An unconstrained run reports 1.0; a run with absurdly tight caps reports
    less on the steps where the caps bite, which is what the progress figure
    shows during a rerun."""
    tr = _small_trainer()
    metrics = tr.train(tmp_path / "free")
    assert np.allclose(metrics["y_trust"], 1.0)

    tight = _small_trainer(y_trust_abs=1e-6)
    tight.y.assign_add(tf.constant(np.full(tight.y.shape, 0.15, dtype="f4")))
    metrics = tight.train(tmp_path / "tight")
    assert float(np.min(metrics["y_trust"])) < 0.5


@pytest.mark.filterwarnings("ignore")
def test_an_anomalous_gradient_skips_the_step(tmp_path):
    """A norm far above the recent median must leave every weight untouched.

    The moments must not move either: if they did, one bad batch would keep
    suppressing (or redirecting) the step for the next hundred updates, which is
    the failure mode beta_2 = 0.99 and AMSGrad are there to shorten.
    """
    tr = _small_trainer()
    # arm the test with a reference far below any real batch's norm
    tr._norm_hist.assign(tf.ones_like(tr._norm_hist))
    tr._norm_n.assign(tr.norm_window)
    w0 = [v.numpy().copy() for v in tr.variables]
    momentum0 = [v.numpy().copy() for v in tr.opt.variables]

    metrics = tr.train(tmp_path)

    assert np.all(np.asarray(metrics["grad_skip"]) == 1.0)
    assert all(np.array_equal(v.numpy(), w) for v, w in zip(tr.variables, w0))
    assert all(np.array_equal(v.numpy(), w) for v, w in zip(tr.opt.variables, momentum0))


@pytest.mark.filterwarnings("ignore")
def test_an_ordinary_gradient_takes_the_step(tmp_path):
    tr = _small_trainer()
    tr._norm_hist.assign(tf.fill(tr._norm_hist.shape, 1e6))
    tr._norm_n.assign(tr.norm_window)
    w0 = [v.numpy().copy() for v in tr.variables]

    metrics = tr.train(tmp_path)

    assert np.all(np.asarray(metrics["grad_skip"]) == 0.0)
    assert any(not np.array_equal(v.numpy(), w) for v, w in zip(tr.variables, w0))


@pytest.mark.filterwarnings("ignore")
def test_the_anomaly_test_waits_for_its_reference():
    """Until the window is full there is no honest median, so nothing is skipped.

    The first norm of a run is far above where it settles (629 against ~40 in
    ``latent_gn_5h``), so an armed test would drop the opening steps.
    """
    tr = _small_trainer()
    assert float(tr._skip_threshold()) == np.inf
    tr._norm_n.assign(tr.norm_window)
    tr._norm_hist.assign(tf.fill(tr._norm_hist.shape, 7.0))
    assert float(tr._skip_threshold()) == pytest.approx(10 * 7.0)

    off = _small_trainer(skip_norm_factor=0.0)
    off._norm_n.assign(off.norm_window)
    assert float(off._skip_threshold()) == np.inf


def test_checkpoints_keep_a_recent_band_and_even_milestones():
    """The two rules, on a run of 10,000 steps checkpointed every 256."""
    from astrotune.train_latent import checkpoints_to_keep

    every, n_steps = 256, 10_000
    steps = list(range(every, n_steps + 1, every))  # 39 checkpoints
    keep = checkpoints_to_keep(steps, n_steps, every, recent=8, milestones=32)

    newest = steps[-1]
    assert {s for s in steps if newest - s <= 8 * every} <= keep  # the recent band
    assert len(keep) <= 8 + 32
    # a milestone from each eighth of the run, so the early history survives
    for i in range(8):
        lo, hi = i * n_steps / 32, (i + 1) * n_steps / 32
        assert any(lo <= s < hi for s in keep), f"nothing kept in slice {i}"
    # and the newest is never dropped
    assert newest in keep


def test_checkpoints_keep_everything_when_the_run_is_short():
    from astrotune.train_latent import checkpoints_to_keep

    steps = [256, 512, 768]  # fewer checkpoints than milestones
    assert checkpoints_to_keep(steps, 800, 256, recent=8, milestones=32) == set(steps)


def test_checkpoint_round_trip_and_retention(tmp_path):
    """A saved state must reload, and the rules must actually delete files."""
    from astrotune.train_latent import LatentTrainConfig, LatentTrainer

    tr = _small_trainer(checkpoint_every_steps=1, keep_recent=2, keep_milestones=1)
    metrics = {"loss": [1.0, 0.9], "step_val": []}
    for step in (2, 4, 6, 8, 10):
        tr.save_state(tmp_path, step, metrics, 0, 1.0, [])

    files = sorted(p.name for p in (tmp_path / "checkpoints").glob("ckpt-*.index"))
    # recent band = the last two saves (8, 10); milestones = the first save (2)
    assert files == ["ckpt-10.index", "ckpt-2.index", "ckpt-8.index"], files
    assert sorted(
        int(p.stem.split("-")[1]) for p in (tmp_path / "checkpoints").glob("state-*.npz")
    ) == [2, 8, 10]

    y_before = tr.y.numpy().copy()
    tr.y.assign(tr.y + 1.0)
    meta = tr.load_state(tmp_path)
    assert meta["step"] == 10
    assert np.allclose(tr.y.numpy(), y_before)
    assert int(tr.opt.iterations) == 0  # the counter is restored as saved


@pytest.mark.filterwarnings("ignore")
def test_sgd_is_available_and_trains(tmp_path):
    """The optimiser can be swapped; SGD takes real steps at a usable rate."""
    from astrotune.train_latent import LatentTrainConfig

    assert LatentTrainConfig().optimizer == "adam"
    tr = _small_trainer(optimizer="sgd", sgd_momentum=0.9)
    y0 = tr.y.numpy().copy()
    metrics = tr.train(tmp_path)
    assert "SGD" in tr.opt.__class__.__name__
    assert np.all(np.isfinite(metrics["loss"]))
    assert not np.allclose(tr.y.numpy(), y0)


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        _small_trainer(optimizer="lamb")
