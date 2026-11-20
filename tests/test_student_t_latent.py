"""Tests for the Student's t data term of the latent-y trainer.

The objective is ``rho = (nu + d) ln(1 + chi2 / nu)`` per observation, whose
curvature ``(nu + d) / (nu + chi2)`` is the weight of the Gauss-Newton step in
``y``; the derivation is in ``notes/gauss_newton_with_students_t.tex``. Three
things are worth testing and are tested here:

* it is the same objective the pipeline was built around in the Gaussian limit,
  which is what keeps the control-point penalties calibrated;
* the step's weight really is the derivative of the loss that is being descended,
  not a quadratic stand-in that drifts away from it;
* and it does what a heavy tail is for -- a wildly discrepant observation pulls
  ``y`` less than it does under the softened chi^2.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sonnet")
pytest.importorskip("tensorflow_probability")
from astropy.table import Table  # noqa: E402

from astrotune import registry  # noqa: E402
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


def _obs(n_cat=2):
    """Observations of ``n_cat`` catalogs that see the same stars, offset slightly."""
    tables = [_table(np.arange(1200), 4, 0) for _ in range(n_cat)]
    for i in range(1, n_cat):
        tables[i]["x"] = tables[0]["x"] + np.array(
            [0.05 * i, 0.1 * i, 0.05 * i, 0.05 * i], dtype="f4"
        )
    return td.build_training_data(tables, seed=1)


def _trainer(obs, ctrl, **overrides):
    from astrotune.train_latent import LatentTrainConfig, LatentTrainer

    settings = dict(
        hidden_size=16, n_hidden=1, batch_size=256, n_epochs=1, max_steps=2,
        plot_every_seconds=None, plot_every_steps=None,
        checkpoint_every_steps=None, agreement_n_stars=256,
    )
    settings.update(overrides)
    return LatentTrainer(obs, ctrl, LatentTrainConfig(**settings))


def _displaced_star(obs, offset, n_obs):
    """The rows of one star with ``n_obs`` observations, the last one displaced."""
    star = np.asarray(obs["star"])
    c = np.asarray(obs["c"])
    for s in np.unique(star):
        rows = np.flatnonzero(star == s)
        if len(rows) == n_obs and len(set(c[rows].tolist())) == n_obs:
            break
    else:  # pragma: no cover - the fixtures always have such stars
        raise AssertionError(f"no star observed by {n_obs} catalogs")
    obs["x"][rows[np.argmax(c[rows])], 0] += offset
    return rows


@pytest.mark.filterwarnings("ignore")
def test_the_gaussian_limit_is_the_plain_chi2():
    """``nu -> inf`` must give back ``chi2`` itself, not a scaled version of it.

    This is the reason the t is written as twice the exact deviance: the pair
    loss averages ``chi2``, the control-point penalties were tuned against that
    scale, and a data term that halved it would silently double every penalty.
    """
    obs, ctrl = _obs()
    tr = _trainer(obs, ctrl, nu=1e6)
    x, xs, c, star = tr.x_obs[:200], tr.xs_obs[:200], tr.c_obs[:200], tr.star_obs[:200]
    x_pred, _ = tr.bij.inverse_with_s(tf.gather(tr.y, star), c=c)

    chi2, d, nu = tr._loss_inputs(x, xs, x_pred, c)
    term = tr._data_term(chi2, d, nu)
    ratio = term.numpy() / np.maximum(chi2.numpy(), 1e-6)
    assert np.allclose(ratio, 1.0, rtol=1e-3), ratio


def test_the_weight_is_the_derivative_of_the_term():
    """The step's weight must be ``d rho / d chi2`` of the loss being descended.

    A weight that is merely proportional is harmless here -- ``w`` cancels out of
    a single observation's normal equations -- but one with the wrong *shape*
    turns the step into an IRLS iteration for a different objective, which is
    what the softened chi^2's ``1/sqrt(1 + (chi2/S)^2)`` was, correctly, for its
    own loss.
    """
    obs, ctrl = _obs()
    tr = _trainer(obs, ctrl, nu=8.0)

    # float64 so that a finite difference of 1e-4 is not swamped by rounding
    chi2 = tf.constant([0.0, 0.5, 4.0, 40.0, 400.0], dtype=tf.float64)
    d = tf.constant([4.0, 4.0, 3.0, 3.0, 4.0], dtype=tf.float64)
    nu = tf.constant([8.0, 8.0, 8.0, 2.0, 8.0], dtype=tf.float64)

    w = tr._data_weight(chi2, d, nu).numpy()
    h = 1e-4
    centre = tr._data_term(chi2, d, nu).numpy()
    numeric = (
        tr._data_term(chi2 + h, d, nu).numpy() - tr._data_term(chi2 - h, d, nu).numpy()
    ) / (2 * h)
    numeric[0] = (tr._data_term(chi2[:1] + h, d[:1], nu[:1]).numpy()[0] - centre[0]) / h

    assert np.allclose(w, numeric, rtol=2e-2), (w, numeric)
    # and it is the expression from the note, (nu + d) / (nu + chi2)
    assert np.allclose(w, (nu.numpy() + d.numpy()) / (nu.numpy() + chi2.numpy()))


def test_a_heavy_tail_downweights_a_bad_observation_more_than_the_softening():
    """The t's weight falls faster than the softening's, at every distance.

    Compared relative to each objective's own weight at ``chi2 = 0``, because the
    t's absolute weight there is ``(nu + d)/nu > 1`` and multiplying a star's
    normal equations by a constant changes nothing -- only the ratio between two
    of its observations does.
    """
    obs, ctrl = _obs()
    t8 = _trainer(obs, ctrl, nu=8.0, data_term="student_t")
    soft = _trainer(obs, ctrl, data_term="softened")

    chi2 = tf.constant([0.5, 10.0, 50.0, 500.0], dtype=tf.float32)
    d = tf.ones_like(chi2)
    nu = tf.fill(tf.shape(chi2), 8.0)
    zero = tf.zeros_like(chi2)

    def relative(tr):
        w = tr._data_weight(chi2, d, nu).numpy()
        return w / tr._data_weight(zero, d, nu).numpy()

    t, s = relative(t8), relative(soft)
    assert np.all(t < s), (t, s)
    assert t[2] < 0.2 < s[2]  # at chi2 = 50: 0.14 against 0.71
    # both fall off as 1/chi2 far out: the difference is the turnover, not the tail
    assert t[1] / t[3] > 10 and s[1] / s[3] > 5


@pytest.mark.filterwarnings("ignore")
def test_a_discrepant_observation_pulls_y_less_under_the_t():
    """The behavioural claim: one 5-sigma outlier, two objectives, same sweeps.

    The star has to be seen by three catalogs. With two, the midpoint of the pair
    is a stationary point of *any* symmetric weight function, so a single
    discrepant observation cannot move ``y`` at all in either objective and the
    test would pass for the wrong reason. With three, the outlier's residual is
    twice the others' at the precision-weighted mean, its weight is smaller, and
    the two objectives part company.
    """
    offset = 0.5  # 5 sigma on a parameter whose quoted error is 0.1
    obs, ctrl = _obs(n_cat=3)
    rows = _displaced_star(obs, offset, n_obs=3)

    t8 = _trainer(obs, ctrl, nu=8.0, data_term="student_t", y_trust_abs=None)
    soft = _trainer(obs, ctrl, data_term="softened", y_trust_abs=None)
    s = int(np.asarray(obs["star"])[rows[0]])
    y_start = t8.y.numpy()[s, 0].copy()

    x = tf.constant(np.asarray(obs["x"])[rows])
    xs = tf.constant(np.asarray(obs["xs"])[rows])
    c = tf.constant(np.asarray(obs["c"])[rows], dtype=tf.int32)
    star = tf.constant(np.asarray(obs["star"])[rows], dtype=tf.int32)

    for _ in range(10):
        t8._gauss_newton_y(x, xs, c, star)
        soft._gauss_newton_y(x, xs, c, star)

    y_t = t8.y.numpy()[s, 0]
    y_s = soft.y.numpy()[s, 0]
    # the displaced observation is the highest catalog's, so both move down from
    # the mean; the t goes further, back towards the measurements it believes
    assert y_t < y_s < y_start, (y_t, y_s, y_start)


def test_the_loss_metric_is_the_data_term_and_the_raw_chi2_is_reported():
    """``chi2`` stays "the data term"; ``chi2_raw`` is the definition-free one."""
    from astrotune.train_latent import BAR_KEYS, METRIC_KEYS

    obs, ctrl = _obs()
    tr = _trainer(obs, ctrl)
    assert "chi2_raw" in tr.metric_keys
    assert "chi2_raw" not in [key for key, _ in BAR_KEYS]
    assert set(METRIC_KEYS) <= set(tr.metric_keys)


def test_nu_comes_from_the_observations_file_when_it_records_one():
    from astrotune.train_latent import LatentTrainConfig, resolve_nu

    cfg = LatentTrainConfig(nu=50.0)
    keys = list(registry.keys()[:3])
    attrs = {"catalogs": ",".join(registry.keys()), "nu": "3.5,9,11"}
    assert np.allclose(resolve_nu(attrs, 3, cfg), [3.5, 9.0, 11.0])
    # a file that records nothing falls back to --nu
    assert np.allclose(resolve_nu({"catalogs": ",".join(keys)}, 3, cfg), 50.0)
    assert np.allclose(resolve_nu(None, 3, cfg), 50.0)
    # and the flag can opt out
    cfg_off = LatentTrainConfig(nu=12.0, nu_from_observations=False)
    assert np.allclose(resolve_nu(attrs, 3, cfg_off), 12.0)


def test_nu_for_other_catalogs_is_refused():
    """The observations index their catalog by position, so a mismatch would
    attach one catalog's tail to another's measurements."""
    from astrotune.train_latent import LatentTrainConfig, resolve_nu

    cfg = LatentTrainConfig()
    attrs = {"catalogs": "some_other_catalog,another", "nu": "3,4"}
    with pytest.raises(ValueError, match="records nu for"):
        resolve_nu(attrs, 2, cfg)
    with pytest.raises(ValueError, match="positive"):
        resolve_nu(None, 2, LatentTrainConfig(nu=0.0))


def test_an_unknown_data_term_is_refused():
    obs, ctrl = _obs()
    with pytest.raises(ValueError, match="data_term must be"):
        _trainer(obs, ctrl, data_term="gaussian")


def test_a_config_from_before_the_t_resumes_as_softened():
    """A saved run that predates the field must not acquire today's default.

    ``LatentTrainConfig(**saved)`` fills a missing key with the *current*
    default, so without this mapping a resume would switch the objective halfway
    through a run and ``_check_resumable`` would see nothing to complain about.
    """
    from astrotune.train_latent import (
        LatentTrainConfig, _check_resumable, _config_from_json,
    )

    saved = {"config": {"freeze_vars": ["embedding"]}}
    loaded = LatentTrainConfig(**_config_from_json(saved["config"]))
    assert loaded.data_term == "softened"
    # so a run resumed with today's defaults is refused, not silently changed
    with pytest.raises(ValueError, match="Cannot change these settings"):
        _check_resumable(saved, LatentTrainConfig())
    # and one resumed with the objective it was trained with is allowed
    _check_resumable(saved, loaded)
