"""Train and evaluate the delta model on stars.

    astrotune-delta-train --run-dir models/my_run

needs the cache written by ``astrotune-apply --run-dir models/my_run``. It

1. builds (or loads) the star table (:mod:`astrotune.delta.stars`) and splits it by star,
2. evaluates the baselines on the validation stars: no added variance, the 0.03
   floor that the merge uses, and one constant ``ln_delta`` per catalog and parameter,
3. trains the network with the star likelihood of :mod:`astrotune.delta.model`
   (the latent-``y`` likelihood, with each star's common value integrated out),
4. measures calibration with and without the learned variances, and writes
   everything to ``<run-dir>/delta/``:

    star_table.npz               the entries of every star (cache)
    diagnostics/                 calibration of the quoted errors (no delta)
    delta_model.*, delta_model_config.json   the trained network
    evaluation/                  baselines, calibration after the fit, figures
    kiel/                        per-catalog Kiel maps of the learned correction
    uncertainty_distribution/    quoted vs calibrated uncertainties, per catalog

Every number in the loss and in the validation metric is **per star**, not per
pair, so a star in nine catalogs counts once.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from .. import registry
from .diagnostics import PARAM_NAMES, SF_THRESHOLDS, diagnose, plot_diagnostics, to_json
from .kiel import make_kiel_figures
from .stars import N_PARAMS, as_prepared, get_star_table, split
from .uncertainties import make_uncertainty_plots

log = logging.getLogger(__name__)


@dataclass
class DeltaConfig:
    hidden_size: int = 64
    n_hidden: int = 3
    epochs: int = 20
    #: Entries per training block. Blocks are cut on star boundaries, so the number
    #: of entries varies by the entries of the last star in each block.
    batch_size: int = 8192
    learning_rate: float = 2e-3
    prior_weight: float = 0.01  # weight of the prior on ln_delta, per measured parameter
    l2_weights: float = 1e-5
    clip_norm: float = 10.0
    seed: int = 0
    # The likelihood is very flat along the direction that trades variance between two
    # catalogs (a star in two catalogs constrains only the sum), so this baseline
    # needs many steps.
    const_epochs: int = 20
    const_learning_rate: float = 0.1
    #: Validation stars used for the NLL (all of them are used for the diagnostics)
    val_max_stars: int = 100_000
    #: Which rows the per-catalog Kiel maps average over: the training stars
    #: (``model``), every prepared row (``all``), or nothing (``none``).
    kiel_sample: str = "model"
    #: Dispersed points per catalog at which the model is nudged to scale 1, add 0.
    #: They are the prior away from the data: 0 switches them off.
    n_control: int = 1024
    control_weight: float = 0.01
    #: Pair chi2 distribution of the validation stars, before and after the fit
    pair_chi2: bool = True
    pair_chi2_max: Optional[int] = 2_000_000
    #: Learn a per-catalog degrees of freedom for a Student's t (False: Gaussian)
    learn_nu: bool = True
    #: Where the prior on nu is Gaussian: in ``raw`` (``nu = 2 + softplus(raw)``, so
    #: ``p(nu)`` rises towards ``nu = 2``) or in ``nu`` itself (a half-Gaussian on
    #: ``nu >= 2``, which vanishes at 2). The mean and sd are read in that space.
    nu_prior_space: str = "raw"
    nu_prior_mean: float = 6.0  # in the unconstrained coordinate: nu = 2 + softplus(6) = 8
    nu_prior_sd: float = 2.0  # 95 % of the prior mass on nu in [4.1, 12.0]
    nu_prior_weight: float = 0.01
    #: Prior that pulls each group's shared variance down to the floor
    shared_prior_weight: float = 0.01
    #: One figure per catalog: quoted against calibrated uncertainties
    uncertainty_plots: bool = True
    #: Fit one shared systematic per group of catalogs that run the same pipeline
    groups: bool = True


def _spec(tf):
    """Input signature: one trace for every block, whatever its length."""
    return {
        "x": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "ls": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "sig": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "J": tf.TensorSpec([None, N_PARAMS, N_PARAMS], tf.float32),
        "c": tf.TensorSpec([None], tf.int32),
        "g": tf.TensorSpec([None], tf.int32),
        "y": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "m": tf.TensorSpec([None, N_PARAMS], tf.bool),
        "seg": tf.TensorSpec([None], tf.int32),
    }


def mean_star_nll(stars, delta_fn, batch_size: int = 131072, max_stars: Optional[int] = None) -> float:
    """Mean negative log likelihood per star over ``stars`` (a proper likelihood)."""
    import tensorflow as tf

    from .model import make_batch, star_nll
    from .stars import block_indices

    if max_stars is not None and len(stars["id"]) > max_stars:
        stars = _first_stars(stars, max_stars)
    n_group, shared = _group_state(delta_fn)
    total, n = 0.0, 0
    nu = delta_fn.nu() if hasattr(delta_fn, "nu") else None
    for idx, seg, n_star in block_indices(stars, None, batch_size):
        nll, _ = star_nll(make_batch(stars, idx, seg), delta_fn, nu=nu, n_group=n_group,
                          shared=shared)
        total += float(tf.reduce_sum(nll))
        n += n_star
    return total / max(n, 1)


def _group_state(model):
    """``(n_group, shared)`` for the hierarchy, or ``(None, None)`` if ungrouped.

    The shared term is only worth its extra level when some group actually has more
    than one catalog; with every group a singleton the flat code path is both
    sufficient and identical. The decision is made from the model's multiplicity, not
    from the current value of tau: a shared variance that starts at zero floor still
    has to be able to grow. Note that ``shared`` is a *snapshot* -- inside a
    ``tf.function`` the caller has to rebuild it from ``model.shared`` (see
    :func:`fit`), or the variable would be baked in as a constant.
    """
    shared_mod = getattr(model, "shared", None)
    if shared_mod is None or not shared_mod.has_shared():
        return None, None
    return int(shared_mod.multi.shape[0]), shared_mod.variance()  # len == n_group


def _first_stars(stars, n_star: int):
    keep = np.zeros(len(stars["id"]), dtype=bool)
    keep[:n_star] = True
    from .stars import subset

    return subset(stars, keep)


def fit(model, stars, cfg: DeltaConfig, epochs: int, lr: float, val=None,
        desc: str = "training", control: Optional[dict] = None) -> list:
    """Minimise the star NLL (+ priors) with Adam and a cosine schedule; returns the history."""
    import tensorflow as tf
    from tensorflow import keras

    from .model import control_penalty, l2_penalty, make_batch, prior_penalty, star_nll
    from .stars import block_indices

    rng = np.random.default_rng(cfg.seed)
    n_star = len(stars["id"])
    steps = max(1, int(np.ceil(stars["off"][-1] / cfg.batch_size)))
    sched = keras.optimizers.schedules.CosineDecay(lr, steps * epochs, alpha=0.01)
    opt = keras.optimizers.Adam(learning_rate=sched)
    variables = list(model.trainable_variables)
    # Static for the whole fit, so the traced step has one branch
    n_group, shared = _group_state(model)

    @tf.function(input_signature=[_spec(tf)])
    def step(batch):
        with tf.GradientTape() as tape:
            # nu and the shared variance must be read *inside* the tape, or their
            # variables are constants to it and never move off their initial value.
            nu = model.nu() if cfg.learn_nu and hasattr(model, "nu") else None
            shared_t = model.shared.variance() if n_group is not None else None
            nll, params = star_nll(batch, model, nu=nu, n_group=n_group, shared=shared_t)
            data = tf.reduce_mean(nll)
            # Every term is kept separately: the history below plots them, and a
            # prior that has quietly taken over the fit is only visible that way.
            prior = tf.cast(cfg.prior_weight * prior_penalty(batch, params), tf.float64)
            l2 = tf.cast(cfg.l2_weights * l2_penalty(model), tf.float64)
            loss = data + prior + l2
            zero = tf.constant(0.0, tf.float64)
            nu_prior = zero
            if getattr(model, "tail", None) is not None:
                # One Gaussian per catalog on the tail weight, in raw or in nu.
                nu_prior = tf.cast(cfg.nu_prior_weight * model.tail.prior_penalty(
                    cfg.nu_prior_mean, cfg.nu_prior_sd, cfg.nu_prior_space), tf.float64)
                loss += nu_prior
            shared_prior = zero
            if getattr(model, "shared", None) is not None:
                # ... and one that pulls each group's shared variance down to the floor
                shared_prior = tf.cast(cfg.shared_prior_weight * model.shared.prior_penalty(),
                                       tf.float64)
                loss += shared_prior
            control_term = zero
            if control is not None and cfg.control_weight:
                # Away from the data the model is pulled to scale 1, add 0: the same
                # prior, evaluated on points the likelihood never sees.
                control_term = tf.cast(cfg.control_weight * control_penalty(control, model),
                                       tf.float64)
                loss += control_term
        grads = tape.gradient(loss, variables)
        # Before clipping: once clipped, every step's norm is capped and the trace
        # stops saying anything about the fit.
        grad_norm = tf.linalg.global_norm(grads)
        grads, _ = tf.clip_by_global_norm(grads, cfg.clip_norm)
        opt.apply_gradients(zip(grads, variables))
        return dict(
            data=data, prior=prior, l2=l2, nu_prior=nu_prior,
            shared_prior=shared_prior, control=control_term, loss=loss,
            grad_norm=tf.cast(grad_norm, tf.float64),
        )

    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n_star)
        parts: dict = {}
        for idx, seg, n in tqdm(block_indices(stars, perm, cfg.batch_size),
                                total=steps, desc=f"{desc} {epoch + 1}/{epochs}", leave=False):
            for key, value in step(make_batch(stars, idx, seg)).items():
                parts.setdefault(key, []).append(float(value))
        rec = dict(epoch=epoch + 1, train_nll=float(np.mean(parts["data"])))
        rec.update({k: float(np.mean(v)) for k, v in parts.items() if k != "data"})
        if val is not None:
            rec["val_nll"] = mean_star_nll(val, model, max_stars=cfg.val_max_stars)
        # The learned parameters at the end of the epoch: where the tail weights and
        # the shared systematics have got to, which the losses alone do not show.
        tail = getattr(model, "tail", None)
        if tail is not None:
            nu_v = np.asarray(tail().numpy(), dtype="f8").ravel()
            if nu_v.size:
                rec["nu_median"] = float(np.median(nu_v))
                rec["nu_min"] = float(nu_v.min())
                rec["nu_max"] = float(nu_v.max())
                rec["nu"] = [float(x) for x in nu_v]
        shared_obj = getattr(model, "shared", None)
        if shared_obj is not None:
            var = np.asarray(shared_obj.variance().numpy(), dtype="f8")
            if var.size:
                sig = np.sqrt(np.maximum(var, 0.0))
                rec["shared_sigma_median"] = float(np.median(sig))
                rec["shared_sigma_max"] = float(sig.max())
        history.append(rec)
        log.info(
            "%s epoch %d/%d: data %.4f%s  loss %.4f  |grad| %.3g",
            desc, epoch + 1, epochs, rec["train_nll"],
            "" if val is None else f", val {rec['val_nll']:.4f}",
            rec["loss"], rec["grad_norm"],
        )
    return history


def plot_training_history(histories: dict, path: Path, title: str = "") -> Path:
    """What each fit did, epoch by epoch: the data term, every prior, and nu.

    ``histories`` maps an arm name (``"const add"``, ``"const scale+add"``,
    ``"network"``) to the list of records :func:`fit` returns. Six panels:

    * the NLL per star against epoch, every arm, with its validation curve dashed;
    * the loss components of the last arm -- the data term against each prior, which
      is how a prior that has quietly taken over the fit becomes visible;
    * the gradient norm before clipping, so a fit that has stalled shows up;
    * every group's ``nu`` per epoch, with the median, because the tail weights are
      learned but the losses barely move once they are near their prior;
    * the shared systematic's sigma per epoch, and
    * the validation minus training NLL, the generalisation gap.
    """
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [(name, hist) for name, hist in histories.items() if hist]
    if not arms:
        return path
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), layout="constrained")
    main, comp, grad, tail, shared, gap = axes.flat

    for name, hist in arms:
        epoch = [r["epoch"] for r in hist]
        main.plot(epoch, [r["train_nll"] for r in hist], label=name)
        val = [(r["epoch"], r["val_nll"]) for r in hist if "val_nll" in r]
        if val:
            main.plot(*zip(*val), ls="--", lw=1.0, color=main.lines[-1].get_color())
    main.set(xlabel="epoch", ylabel="NLL per star",
             title="data term (solid) and validation (dashed)")
    main.legend(fontsize=7)

    name, hist = arms[-1]
    epoch = np.asarray([r["epoch"] for r in hist], dtype="f8")
    for key, label in (("loss", "total"), ("data", "data term"), ("prior", "prior"),
                       ("l2", "weight decay"), ("nu_prior", "prior on nu"),
                       ("shared_prior", "prior on the shared term"),
                       ("control", "control points")):
        values = np.asarray([r.get(key, 0.0) for r in hist], dtype="f8")
        if np.any(np.abs(values) > 0):
            comp.semilogy(epoch, np.maximum(np.abs(values), 1e-12), label=label, lw=1.2)
    comp.set(xlabel="epoch", ylabel="mean |term|", title=f"{name}: loss components")
    comp.legend(fontsize=7)

    for name, hist in arms:
        grad.plot([r["epoch"] for r in hist],
                  [max(r.get("grad_norm", 0.0), 1e-12) for r in hist], label=name)
    grad.set(xlabel="epoch", ylabel="|grad| before clipping", title="gradient norm", yscale="log")
    grad.legend(fontsize=7)

    nu_rows = [r for r in hist if "nu" in r]
    if nu_rows:
        nu = np.asarray([r["nu"] for r in nu_rows], dtype="f8")  # (epochs, n_group)
        e = np.asarray([r["epoch"] for r in nu_rows], dtype="f8")
        tail.plot(e, nu, color="0.7", lw=0.7)
        tail.plot(e, np.median(nu, axis=1), color="C0", lw=2.0, label="median")
        tail.plot(e, nu.min(axis=1), color="C3", lw=1.0, ls=":", label="min")
        tail.plot(e, nu.max(axis=1), color="C3", lw=1.0, ls=":", label="max")
        tail.axhline(2.0, color="0.4", ls="--", lw=0.8)  # the floor: nu = 2 + softplus
        tail.set(xlabel="epoch", ylabel="nu", title="tail weight per group (learned)")
        tail.legend(fontsize=7)
    else:
        tail.set_axis_off()

    rows = [r for r in hist if "shared_sigma_median" in r]
    if rows:
        shared.plot([r["epoch"] for r in rows], [r["shared_sigma_median"] for r in rows],
                    color="C2", label="median")
        shared.plot([r["epoch"] for r in rows], [r["shared_sigma_max"] for r in rows],
                    color="C2", lw=1.0, ls=":", label="max")
        shared.set(xlabel="epoch", ylabel="shared sigma", title="shared systematic", yscale="log")
        shared.legend(fontsize=7)
    else:
        shared.set_axis_off()

    plotted = False
    for name, hist in arms:
        pairs = [(r["epoch"], r["val_nll"] - r["train_nll"]) for r in hist
                 if "val_nll" in r and "train_nll" in r]
        if pairs:
            gap.plot(*zip(*pairs), label=name)
            plotted = True
    if plotted:
        gap.axhline(0.0, color="0.5", lw=0.8)
        gap.set(xlabel="epoch", ylabel="validation - training NLL",
                title="generalisation gap")
        gap.legend(fontsize=7)
    else:
        gap.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def nu_prior_curve(nu, mean: float, sd: float, space: str = "raw"):
    """``p(nu)`` for the prior used on the tail weight.

    With ``space="raw"`` the prior is Gaussian in the unconstrained coordinate and

        p(nu) = N(raw(nu); mean, sd) * |d raw / d nu|,

    with ``raw(nu) = log(exp(nu - 2) - 1)`` and ``d raw / d nu = 1 / (1 -
    exp(-(nu - 2)))``. The Jacobian is what the figure is for: it makes ``p(nu)``
    diverge as ``nu -> 2``, because the softplus squeezes an ever longer stretch of
    ``raw`` into a shrinking ``d nu`` there.

    With ``space="nu"`` the prior is the half-Gaussian

        p(nu) ~ exp(-(nu - mean)^2 / 2 sd^2)   on nu >= 2,  0 below,

    which has no such spike: it vanishes at ``nu = 2`` and is a plain Gaussian
    above it.
    """
    nu = np.atleast_1d(np.asarray(nu, dtype="f8"))
    t = np.maximum(nu - 2.0, 1e-12)  # nu - 2, the softplus output
    if space == "nu":
        dens = np.exp(-0.5 * ((nu - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        return np.where(t > 0.0, dens, 0.0)
    if space != "raw":
        raise ValueError(f"unknown nu prior space {space!r}; use 'raw' or 'nu'")
    raw = np.log(np.expm1(np.minimum(t, 700.0)))  # inverse softplus
    dens = np.exp(-0.5 * ((raw - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return dens / (1.0 - np.exp(-t))  # the Jacobian d raw / d nu


def plot_nu_prior(nu_values, keys: Sequence[str], prior_mean: float, prior_sd: float,
                  path: Path, title: str = "", space: str = "raw") -> Path:
    """The prior on ``nu``, with each group's learned ``nu`` marked on it.

    The point of the figure is to show in one glance how much of each group's
    ``nu`` is the prior: a dot near the peak is a parameter the prior set, a dot
    out on the flank is one the data moved. The curves differ only in width and
    centre, so runs with different priors can be compared page for page.
    """
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .. import registry

    nu_values = np.asarray(nu_values, dtype="f8").ravel()
    # One label per group, in registry-key form; a group of two catalogs is named
    # for both, since they share the value.
    gidx = registry.group_index(keys)
    labels = []
    for g in range(len(nu_values)):
        members = [k for k, gi in zip(keys, gidx) if gi == g]
        labels.append("+".join(members) if members else str(g))

    prior_nu = (float(prior_mean) if space == "nu"
                else 2.0 + float(np.logaddexp(0.0, prior_mean)))  # nu at the prior mean
    # p(nu) diverges at nu = 2 under the raw-space prior (integrably), so the curve is
    # cut where it is still small in every prior we use rather than at the singularity.
    lo, hi = 2.5, max(nu_values.max(), prior_nu + 3.5 * prior_sd, 10.0) * 1.08
    grid = np.linspace(lo, hi, 1200)
    curve = nu_prior_curve(grid, prior_mean, prior_sd, space)
    peak = float(curve.max())

    fig_h, fs = 9.0, 6.5
    fig, ax = plt.subplots(figsize=(12, fig_h), layout="constrained")

    # Nearly all the groups sit within a tenth of nu of the prior mean, so their
    # labels cannot stand side by side above their own dots. They are instead
    # spread rightwards into a row (never leftwards, so the order still reads off
    # the axis) and drawn from one common baseline, each with a leader line back to
    # its dot.
    gap = (fs / 72.0) / (12.0 * 0.85) * (hi - lo) * 1.25  # a label's width, in nu
    label_x = {}
    last = -np.inf
    for i in np.argsort(nu_values):
        x = max(nu_values[i], last + gap)
        label_x[i] = x
        last = x
    # Height of the tallest label as a fraction of the axes, from its point size.
    hf = (max(len(s) for s in labels) * 0.62 + 1.0) * fs / 72.0 / (fig_h * 0.82)
    base = peak * 1.05
    top = base / (1.0 - hf)

    ax.plot(grid, curve, color="C0", lw=1.6)
    ax.fill_between(grid, 0.0, curve, color="C0", alpha=0.10)
    ax.axvline(prior_nu, color="C0", ls="--", lw=0.9, alpha=0.7)
    ax.annotate(f"prior: $\\nu$ = {prior_nu:.1f} $\\pm$ {prior_sd:.1f}",
                xy=(prior_nu, peak * 0.42), xytext=(-11, 0),
                textcoords="offset points", fontsize=8, color="C3", rotation=90,
                va="center")

    for i in np.argsort(nu_values):
        x = nu_values[i]
        y = float(nu_prior_curve(x, prior_mean, prior_sd, space)[0])
        ax.plot([x, label_x[i]], [y, base], color="0.6", lw=0.5, zorder=1)
        ax.plot([x], [y], marker="o", ms=4.5, color="C3", zorder=3)
        ax.text(label_x[i], base, labels[i], rotation=90, fontsize=fs,
                ha="center", va="bottom", zorder=4)

    ax.set(xlabel=r"$\nu$ (degrees of freedom)", ylabel=r"$p(\nu)$",
           xlim=(lo, max(hi, last + 2 * gap)), ylim=(0.0, top),
           title=title or "prior on nu")
    ax.grid(alpha=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_learned_delta(model, stars, labels: Sequence[str], path: Path) -> Path:
    """Median effective error inflation against the catalog's own Teff coordinate.

    The y axis is ``sigma_out / sigma_in = sqrt(scale + add/C)``: 1 means the quoted
    error is left alone, below 1 that the catalog is being shrunk, above 1 that it is
    being widened. Dimensionless, so catalogs of very different precision share one
    axis (the additive sigma alone does not).
    """
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..training_plots import catalog_colors
    from .covariance import entry_cov_np, sigma_out_np
    from .model import model_shared, numpy_delta_fn

    fn = numpy_delta_fn(model)
    x, ls, c, g, m = stars["x"], stars["ls"], stars["c"], stars["g"], stars["m"]
    ln_scale, ln_add = fn(x, ls, g)
    shared = model_shared(model)
    total = entry_cov_np(stars["sig"], stars["J"], ln_scale, ln_add, measured=m)
    if shared is not None:
        ii = np.arange(N_PARAMS)
        total[:, ii, ii] += shared[g]
    ratio = sigma_out_np(total) / sigma_out_np(
        entry_cov_np(stars["sig0"], stars["J"], measured=m))
    colors = catalog_colors(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True)
    edges = np.quantile(x[:, 0], np.linspace(0, 1, 13))
    which = np.clip(np.digitize(x[:, 0], edges[1:-1]), 0, 11)
    for p, ax in enumerate(axes.flat):
        for k, lab in enumerate(labels):
            sel = (c == k) & m[:, p]
            if sel.sum() < 2000:
                continue
            xc, yc = [], []
            for b in range(12):  # twelve Teff bins, not one pass per entry
                s = which[sel] == b
                if s.sum() >= 100:
                    xc.append(np.median(x[sel][s, 0]))
                    yc.append(np.median(ratio[sel][s, p]))
            ax.plot(xc, yc, "-", color=colors[k], label=lab, lw=1.2)
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set(title=PARAM_NAMES[p], ylabel=r"median $\sigma_{\rm out}/\sigma_{\rm in}$")
        ax.grid(True, alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel(r"catalog's own $5\ln(T_{\rm eff}/5000\,{\rm K})$")
        ax.invert_xaxis()
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def compare(results: Dict[str, dict], nll: Dict[str, float]) -> str:
    """A table of the headline numbers for several covariance models."""
    head = (f"{'model':22s}{'val NLL/star':>13s}{'median PIT':>11s}"
            + "".join(f"{'sf<' + format(t, 'g'):>10s}" for t in SF_THRESHOLDS)
            + "".join(f"{n:>9s}" for n in PARAM_NAMES) + "   (scale ratio)")
    lines = [head]
    for name, r in results.items():
        p = r["pooled"]
        lines.append(
            f"{name:22s}{nll[name]:13.4f}{p['median_pit']:11.3f}"
            + "".join(f"{f:10.2%}" for f in p["frac_sf"])
            + "".join(f"{s:9.2f}" for s in p["scale_ratio"])
        )
    lines.append(f"{'(nominal)':22s}{'':13s}{0.5:11.3f}" + "".join(f"{t:10.2%}" for t in SF_THRESHOLDS))
    return "\n".join(lines)


def run(run_dir: Path, out_dir: Optional[Path] = None, cfg: Optional[DeltaConfig] = None,
        rebuild: bool = False, min_entries: int = 1000) -> dict:
    import tensorflow as tf

    from .model import (ConstDelta, ConstScaleAdd, ScaleAddNetwork, control_points, fixed_delta,
                        model_nu, model_shared, numpy_delta_fn, save_delta_model)

    cfg = cfg or DeltaConfig()
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta"
    ev_dir = out_dir / "evaluation"
    ev_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(cfg.seed)

    stars, keys = get_star_table(run_dir, out_dir, rebuild)
    labels = [registry.get_spec(k).label for k in keys]
    n_cat = len(keys)
    # The model's category is the group: catalogs that run the same pipeline on the
    # same spectra share their correction (and, through the hierarchy, their errors).
    multiplicity = (registry.group_multiplicity(keys) if cfg.groups
                    else tuple([1] * n_cat))
    n_group = len(multiplicity)
    group_labels = (registry.group_labels(keys) if cfg.groups else tuple(labels))
    train, val = split(stars)
    ctrl = control_points(stars, keys, cfg.n_control, cfg.seed) if cfg.n_control else None
    del stars
    log.info("%d training stars (%d entries), %d validation stars (%d entries)",
             len(train["id"]), int(train["off"][-1]), len(val["id"]), int(val["off"][-1]))
    log.info("%d groups for %d catalogs: %s", n_group, n_cat,
             [f"{g} ({m} catalogs)" for g, m in zip(group_labels, multiplicity) if m > 1] or "none")
    if ctrl is not None:
        log.info("%d control points (%d per catalog, weight %.3g)",
                 len(ctrl["c"]), cfg.n_control, cfg.control_weight)

    # The reference arm is the covariance the pipeline propagates *today* -- quoted,
    # scaled by the catalog's UNC_SCALES and floored by ERROR_FLOORS -- while the model
    # is fitted on the raw quoted sigma, so the two live on different tables.
    ref = as_prepared(val)
    models = {"quoted (as prepared)": fixed_delta(0.0)}
    tables_for = {"quoted (as prepared)": ref}
    const_add = ConstDelta(n_group, multiplicity)
    hist_const = fit(const_add, train, cfg, cfg.const_epochs, cfg.const_learning_rate, val,
                     desc="const add", control=ctrl)
    models["constant add"] = const_add

    const = ConstScaleAdd(n_group, multiplicity)
    hist_const_sa = fit(const, train, cfg, cfg.const_epochs, cfg.const_learning_rate, val,
                        desc="const scale+add", control=ctrl)
    models["constant scale+add"] = const

    net = ScaleAddNetwork(n_group, cfg.hidden_size, cfg.n_hidden, multiplicity)
    hist = fit(net, train, cfg, cfg.epochs, cfg.learning_rate, val, desc="network", control=ctrl)
    models["network scale+add"] = net

    save_delta_model(out_dir, net, dict(
        n_cat=n_group, hidden_size=cfg.hidden_size, n_hidden=cfg.n_hidden, keys=list(keys),
        multiplicity=list(multiplicity), group_labels=list(group_labels),
        model_type="scale_add", config=dataclasses.asdict(cfg), val_history=hist,
    ))
    # Before the diagnostics, so the training traces survive a failure in them.
    plot_training_history(
        {"const add": hist_const, "const scale+add": hist_const_sa, "network": hist},
        ev_dir / "training_history.png",
        title=f"{run_dir.name}: delta model training",
    )
    log.info("wrote %s", ev_dir / "training_history.png")

    nll, results, nus, shareds = {}, {}, {}, {}
    for name, fn in models.items():
        t = tables_for.get(name, val)
        nll[name] = mean_star_nll(t, fn, max_stars=cfg.val_max_stars)
        np_fn = numpy_delta_fn(fn) if hasattr(fn, "trainable_variables") else _fixed_np(fn)
        # With a learned tail weight the null of every chi2 diagnostic is an F
        # distribution, so the diagnostics need nu as well as the covariance.
        nus[name] = model_nu(fn) if hasattr(fn, "trainable_variables") else None
        shareds[name] = model_shared(fn) if hasattr(fn, "trainable_variables") else None
        results[name] = diagnose(t, n_cat, np_fn, min_entries, nu=nus[name],
                                 group_cov=shareds[name], group_map=_group_map(keys))
        log.info("%s: validation NLL per star %.4f", name, nll[name])
    table = compare(results, nll)
    log.info("\n%s", table)
    (ev_dir / "comparison.txt").write_text(table + "\n")
    # The scale/add table says what the two corrections actually did, per catalog, at
    # the catalog's own median quoted error - the quantity the merge will see.
    sa_table = scale_add_table(models, val, labels)
    log.info("\n%s", sa_table)
    (ev_dir / "scale_add.txt").write_text(sa_table + "\n")
    nu_table = format_nu(nus, group_labels)
    if nu_table:
        log.info("\n%s", nu_table)
        (ev_dir / "nu.txt").write_text(nu_table + "\n")
    shared_table = format_shared(shareds, group_labels)
    if shared_table:
        log.info("\n%s", shared_table)
        (ev_dir / "shared.txt").write_text(shared_table + "\n")
    # One line per catalog: what the fit did to it, in the four numbers that describe
    # an entry's covariance (size, sigma scale, additive sigma, tail weight, volume).
    cat_table = per_catalog_table(net, val, keys)
    log.info("\n%s", cat_table)
    (ev_dir / "per_catalog.txt").write_text(cat_table + "\n")
    # The same numbers as a picture: does the group that asks for heavy tails also
    # have its covariance inflated (or shrunk) most?
    plot_nu_det_c(catalog_summary(net, val, keys), ev_dir / "nu_det_c.png",
                  title=f"{run_dir.name}: nu against the volume of the corrected covariance")
    # The additive-only constant model is the maximum-likelihood three-cornered hat:
    # one excess variance per catalog and parameter that makes the entries agree.
    const_sigma = np.exp(0.5 * const_add.raw.numpy())
    np.savez(ev_dir / "const_delta.npz", sigma=const_sigma, keys=list(keys),
             scale=const.scale_sigma(), add_sigma=const.add_sigma(),
             shared=model_shared(net), multiplicity=np.asarray(multiplicity),
             group_labels=np.asarray(group_labels, dtype=object))
    with open(ev_dir / "comparison.json", "w") as f:
        json.dump(to_json(dict(nll=nll, results={k: {kk: vv for kk, vv in v.items() if kk != "entries"}
                                                   for k, v in results.items()},
                               nus=nus, shared=shareds, keys=list(keys),
                               group_labels=list(group_labels),
                               multiplicity=list(multiplicity),
                               const_sigma=const_sigma, const_scale=const.scale_sigma(),
                               const_add=const.add_sigma(), history=hist,
                               const_history=hist_const, const_scale_add_history=hist_const_sa)),
                  f, indent=1)
    plot_diagnostics(results["network scale+add"], labels, ev_dir, "_network")
    plot_diagnostics(results["quoted (as prepared)"], labels, ev_dir, "_none")
    plot_learned_delta(net, val, labels, ev_dir / "learned_delta.png")
    # The prior on nu with every group's learned value on it: how much of each nu
    # was the prior, and how much the data.
    if nus.get("network scale+add") is not None:
        plot_nu_prior(nus["network scale+add"], keys, cfg.nu_prior_mean, cfg.nu_prior_sd,
                      ev_dir / "nu_prior.png",
                      title=(f"{run_dir.name}: prior on nu "
                             f"({cfg.nu_prior_mean:g} $\\pm$ {cfg.nu_prior_sd:g}"
                             + (" in nu)" if cfg.nu_prior_space == "nu" else " in raw)")),
                      space=cfg.nu_prior_space)
    if cfg.pair_chi2:
        # Two observations of one star: the difference's covariance is S_a + S_b with
        # nothing estimated in between, so the chi2 distribution is a direct test.
        from .diagnostics import pair_chi2, pair_chi2_text, plot_pair_chi2

        before = pair_chi2(ref, None, max_pairs=cfg.pair_chi2_max)
        after = pair_chi2(val, numpy_delta_fn(net), max_pairs=cfg.pair_chi2_max,
                          nu=nus["network scale+add"], group_cov=shareds["network scale+add"])
        text = pair_chi2_text(before, after)
        log.info("\n%s", text)
        (ev_dir / "pair_chi2.txt").write_text(text + "\n")
        plot_pair_chi2(before, after, ev_dir / "pair_chi2.png")
    if cfg.kiel_sample != "none":
        # Where in the Kiel diagram the correction sits, one page pair per catalog.
        make_kiel_figures(run_dir, out_dir / "kiel", model=net, sample=cfg.kiel_sample, keys=keys)
    if cfg.uncertainty_plots:
        # Quoted against calibrated uncertainties: is the corrected error plausible?
        make_uncertainty_plots(run_dir, out_dir / "uncertainty_distribution", model=net,
                               sample="train", keys=keys)
    return dict(nll=nll, table=table, scale_add=sa_table)


def _group_map(keys) -> np.ndarray:
    """Group index of every catalog, for mapping ``stars["c"]`` to its group."""
    return np.asarray(registry.group_index(keys), dtype="i4")


def format_shared(shareds: Dict[str, Optional[np.ndarray]], labels: Sequence[str]) -> str:
    """The fitted shared sigma per group and parameter, one column per trained arm.

    This is the part of a group's error that its catalogs have *in common*: it does
    not shrink when they are combined, and it has to be added to each catalog's own
    covariance before the merge. Groups of one catalog have none and are omitted.
    """
    arms = [(name, s) for name, s in shareds.items() if s is not None]
    if not arms:
        return ""
    # Groups with a shared term in any arm; the singletons are exactly zero.
    multi = [i for i in range(len(arms[0][1]))
             if any(np.any(s[i] > 0) for _, s in arms)]
    if not multi:
        return ""
    width = max(len(labels[i]) for i in multi)
    head = f"{'shared sigma':14s}{'group':{width + 2}s}" + "".join(
        f"{n:>16s}" for n, _ in arms)
    lines = [head, "-" * len(head)]
    for i in multi:
        for p, pname in enumerate(PARAM_NAMES):
            lines.append(
                f"{pname:14s}{(labels[i] if p == 0 else ''):{width + 2}s}"
                + "".join(f"{np.sqrt(max(s[i, p], 0.0)):16.4f}" for _, s in arms))
    return "\n".join(lines)


def format_nu(nus: Dict[str, Optional[np.ndarray]], labels: Sequence[str]) -> str:
    """The fitted degrees of freedom per catalog, one column per trained arm."""
    arms = [(name, nu) for name, nu in nus.items() if nu is not None]
    if not arms:
        return ""
    name_w = max(len(n) for n in ["catalog"] + list(labels))
    head = f"{'catalog':{name_w}s}" + "".join(f"{n:>22s}" for n, _ in arms)
    lines = [head, "-" * len(head)]
    for k, lab in enumerate(labels):
        lines.append(f"{lab:{name_w}s}" + "".join(f"{nu[k]:22.1f}" for _, nu in arms))
    return "\n".join(lines)


def catalog_summary(model, stars, keys: Sequence[str], chunk: int = 65_536) -> dict:
    """Per catalog: the size, the two learned terms, ``nu``, and the volume change.

    The numbers behind :func:`per_catalog_table` and the ``nu``-against-volume plot,
    so the two cannot disagree. ``det`` is the median over the catalog's entries of
    ``(ln det(S + diag(tau^2)) - ln det C_m) / n_measured`` -- the change in the log
    determinant of the covariance the merge will use, per measured parameter. The
    division is what makes a catalog that measures three parameters comparable with
    one that measures four: the total volume change of a 4-parameter covariance is
    four times the average per-parameter change, so leaving it out would rank the
    catalogs by how many parameters they happen to report.
    """
    from .covariance import entry_cov_np
    from .model import model_nu, model_shared, numpy_delta_fn

    fn = numpy_delta_fn(model)
    nu = model_nu(model)
    shared = model_shared(model)
    n_cat, n_par = len(keys), N_PARAMS
    n_ent = int(stars["off"][-1])
    s_val = np.zeros((n_ent, n_par), dtype="f4")  # sigma scale
    a_val = np.zeros((n_ent, n_par), dtype="f4")  # additive sigma, shared included
    dlndet = np.full(n_ent, np.nan)
    for lo in range(0, n_ent, chunk):
        sl = slice(lo, min(lo + chunk, n_ent))
        m = stars["m"][sl]
        ln_scale, ln_add = fn(stars["x"][sl], stars["ls"][sl], stars["g"][sl])
        s = np.exp(0.5 * ln_scale.astype("f8"))
        add = np.exp(ln_add.astype("f8"))
        if shared is not None:
            add = add + np.asarray(shared, dtype="f8")[stars["g"][sl]]
        a_val[sl] = np.sqrt(add)
        s_val[sl] = s
        # The determinant over the parameters the entry measures (a placeholder axis of
        # 1e9 would otherwise dominate it), grouped by mask so the subset is exact.
        new = entry_cov_np(stars["sig"][sl], stars["J"][sl], ln_scale, ln_add, measured=m)
        old = entry_cov_np(stars["sig0"][sl], stars["J"][sl], measured=m)
        ii = np.arange(N_PARAMS)
        if shared is not None:
            new[:, ii, ii] += np.asarray(shared, dtype="f8")[stars["g"][sl]]
        masks, which = np.unique(m, axis=0, return_inverse=True)
        for j, mask in enumerate(masks):
            rows = which == j
            if not mask.any() or not rows.any():
                continue
            dlndet[lo + np.flatnonzero(rows)] = (
                np.linalg.slogdet(new[rows][:, mask][:, :, mask])[1]
                - np.linalg.slogdet(old[rows][:, mask][:, :, mask])[1]) / float(mask.sum())

    count = np.zeros(n_cat, dtype="i8")
    scale = np.full((n_cat, n_par), np.nan)
    add = np.full((n_cat, n_par), np.nan)
    det = np.full(n_cat, np.nan)
    nu_cat = np.full(n_cat, np.nan)
    for k in range(n_cat):
        in_cat = stars["c"] == k
        count[k] = int(in_cat.sum())
        for p in range(n_par):
            sel = in_cat & stars["m"][:, p]
            if sel.any():
                scale[k, p] = float(np.median(s_val[sel, p]))
                add[k, p] = float(np.median(a_val[sel, p]))
        de = dlndet[in_cat & np.isfinite(dlndet)]
        if de.size:
            det[k] = float(np.median(de))
        if nu is not None and in_cat.any():
            nu_cat[k] = float(nu[stars["g"][np.argmax(in_cat)]])
    return dict(keys=tuple(keys), n=count, scale=scale, add=add, det=det, nu=nu_cat)


def _label_positions(points, sizes, bounds, pad: float = 0.05, passes: int = 15):
    """Place a horizontal label near each point by minimising a layout energy.

    Everything is in inches, so distances mean the same thing in both directions.
    ``points`` is ``(n, 2)``, ``sizes`` is ``(n, 2)`` of ``(width, height)``, and
    ``bounds`` is ``(x0, x1, y0, y1)`` of the point cloud. The energy sums

    * overlapping labels, weighted by the overlapping area;
    * a label sitting on top of a point (any point, its own included);
    * a label that leaves the box the points occupy;
    * two leader lines crossing;
    * and, weakly, the distance from a label to its own point, which is what keeps a
      free label next to its dot instead of merely somewhere legal.

    The labels are placed by coordinate descent over a fixed set of candidate
    positions around each dot -- eight directions at three distances -- started from
    a few different configurations, since the energy is not convex.
    """
    n = len(points)
    half = sizes / 2.0
    dirs = np.array([[0, 1], [1, 1], [1, 0], [1, -1],
                     [0, -1], [-1, -1], [-1, 0], [-1, 1]], dtype="f8")
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    gaps = np.array([0.05, 0.13, 0.24])
    # Candidate box centres: just outside the dot in that direction, so the leader
    # line leaves the dot and meets the near face of the box.
    extent = (np.abs(dirs[:, 0])[None, :] * half[:, None, 0]
              + np.abs(dirs[:, 1])[None, :] * half[:, None, 1])  # (n, n_dir)
    cand = np.empty((n, len(dirs) * len(gaps), 2))
    for gi, g in enumerate(gaps):
        for di in range(len(dirs)):
            col = gi * len(dirs) + di
            cand[:, col] = points + dirs[di] * (g + extent[:, di])[:, None]
    w, h = sizes[:, 0], sizes[:, 1]

    def leader_ends(centres):
        """Where each leader meets its label box (the nearest point on the box)."""
        q = np.empty_like(centres)
        q[:, 0] = np.clip(points[:, 0], centres[:, 0] - w / 2, centres[:, 0] + w / 2)
        q[:, 1] = np.clip(points[:, 1], centres[:, 1] - h / 2, centres[:, 1] + h / 2)
        return q

    def energy(a):
        cen = cand[np.arange(n), a]
        dx = np.abs(cen[:, None, 0] - cen[None, :, 0]) - (w[:, None] + w[None, :]) / 2
        dy = np.abs(cen[:, None, 1] - cen[None, :, 1]) - (h[:, None] + h[None, :]) / 2
        overlap = np.clip(pad - dx, 0, None) * np.clip(pad - dy, 0, None)
        e = 60.0 * np.triu(overlap, 1).sum()
        # a label covering a dot: its own is excluded only when the box is clear of
        # it, which the candidates already ensure
        inx = np.abs(points[None, :, 0] - cen[:, None, 0]) < w[:, None] / 2 + 0.02
        iny = np.abs(points[None, :, 1] - cen[:, None, 1]) < h[:, None] / 2 + 0.02
        e += 40.0 * (inx & iny).sum()
        outside = (np.clip(bounds[0] - (cen[:, 0] - w / 2), 0, None)
                   + np.clip((cen[:, 0] + w / 2) - bounds[1], 0, None)
                   + np.clip(bounds[2] - (cen[:, 1] - h / 2), 0, None)
                   + np.clip((cen[:, 1] + h / 2) - bounds[3], 0, None))
        e += 12.0 * outside.sum()
        q = leader_ends(cen)
        # Two segments cross when each has the other's endpoints on opposite sides of
        # its own line; the transpose gives the second segment's test.
        def cross(o, a1, b):
            return ((a1[None, :, 0] - o[:, None, 0]) * (b[:, None, 1] - o[:, None, 1])
                    - (a1[None, :, 1] - o[:, None, 1]) * (b[:, None, 0] - o[:, None, 0]))
        straddle = (cross(points, q, points) * cross(points, q, q)) < 0
        e += 4.0 * float((straddle & straddle.T).sum()) / 2.0
        e += 1.0 * float(np.linalg.norm(cen - points, axis=1).sum())
        return e

    best_a, best_e = None, np.inf
    for start in (0, 2, 4, 6):  # four of the eight directions
        a = np.full(n, start * len(gaps) + 1)
        e = energy(a)
        for _ in range(passes):
            moved = False
            for i in range(n):
                cur, best_i, best_e_i = a[i], a[i], e
                for c in range(cand.shape[1]):
                    if c == cur:
                        continue
                    a[i] = c
                    ec = energy(a)
                    if ec < best_e_i - 1e-9:
                        best_i, best_e_i = c, ec
                a[i] = best_i
                moved |= best_i != cur
                e = best_e_i
            if not moved:
                break
        if e < best_e:
            best_a, best_e = a.copy(), e
    return cand[np.arange(n), best_a], leader_ends(cand[np.arange(n), best_a])


def plot_nu_det_c(summary: dict, path: Path, title: str = "") -> Path:
    """``nu`` against the volume of the corrected covariance, one dot per catalog.

    ``nu`` is one number per group and ``dln det C`` is one per catalog, so the two
    members of a group sit at the same ``nu``. The panel asks whether the catalogs
    that ask for heavy tails are the ones whose covariance is being inflated or
    shrunk most: they need not be related, since ``nu`` describes the shape of the
    residual distribution and ``dln det C`` its width.
    """
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(summary["nu"], dtype="f8")
    y = np.asarray(summary["det"], dtype="f8")
    labels = [str(k) for k in summary["keys"]]
    keep = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if not keep.size:
        return path

    fig_w, fig_h = 12.0, 8.0
    fs = 6.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    left, right, bottom, top = 0.075, 0.995, 0.085, 0.935
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax_w, ax_h = fig_w * (right - left), fig_h * (top - bottom)

    xr = float(x[keep].max() - x[keep].min()) or 1.0
    yr = float(y[keep].max() - y[keep].min()) or 1.0
    xlim = (float(x[keep].min()) - 0.06 * xr, float(x[keep].max()) + 0.10 * xr)
    ylim = (float(y[keep].min()) - 0.15 * yr, float(y[keep].max()) + 0.15 * yr)
    ax.set(xlim=xlim, ylim=ylim)

    def to_inch(px, py):
        return np.stack([(px - xlim[0]) / (xlim[1] - xlim[0]) * ax_w,
                         (py - ylim[0]) / (ylim[1] - ylim[0]) * ax_h], axis=-1)

    def to_data(ix, iy):
        return ((ix / ax_w) * (xlim[1] - xlim[0]) + xlim[0],
                (iy / ax_h) * (ylim[1] - ylim[0]) + ylim[0])

    pts = to_inch(x[keep], y[keep])
    # A label's footprint, from its point size; 0.58 em per character is right for
    # DejaVu Sans at this size to within a few per cent.
    sizes = np.stack([0.58 * fs * np.array([len(labels[j]) for j in keep]) / 72.0,
                      np.full(len(keep), 1.35 * fs / 72.0)], axis=-1)
    lo_pt = pts.min(axis=0)
    hi_pt = pts.max(axis=0)
    centres, ends = _label_positions(pts, sizes, (lo_pt[0], hi_pt[0], lo_pt[1], hi_pt[1]))

    for k, j in enumerate(keep):
        (lx, ly), (qx, qy) = to_data(centres[k, 0], centres[k, 1]), to_data(ends[k, 0],
                                                                           ends[k, 1])
        ax.plot([x[j], qx], [y[j], qy], color="0.6", lw=0.5, zorder=1)
        ax.plot([x[j]], [y[j]], marker="o", ms=5, color="C3", zorder=3)
        ax.text(lx, ly, labels[j], fontsize=fs, ha="center", va="center", zorder=4)

    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set(xlabel=r"$\nu$ (group's degrees of freedom)",
           ylabel=r"$\Delta \ln \det C\,/\,n_{\rm measured}$ (median over the catalog)",
           title=title or "nu against the volume of the corrected covariance")
    ax.grid(alpha=0.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def per_catalog_table(model, stars, keys: Sequence[str], chunk: int = 65_536) -> str:
    """One row per catalog: size, the two learned terms, the tail weight, and the effect.

    The rows are the *validation* entries of each catalog (the sample every other
    number in ``evaluation/`` is measured on). ``median scale`` is, per parameter, the
    median of the sigma scale ``exp(ln_scale / 2)`` over those entries that measure it
    (1 = the quoted error is left alone); ``median add*`` is the same for the additive
    sigma *including the group's shared variance* (the floor is 0.003, so a value above
    it is the systematic the catalog shares with the rest of its group); ``nu`` is the
    group's degrees of freedom; and ``median dln det/p*`` is the median change in the
    log determinant of the total entry covariance ``S + diag(tau^2)``,
    ``(ln det - ln det C_m) / n_measured`` over the measured submatrix -- how much
    volume the correction adds or removes *per parameter*, which is the one number
    that combines all three terms and means the same thing for a catalog reporting
    three parameters as for one reporting four (at scale 0.35 with nothing additive,
    ``2 ln 0.35 = -2.1`` either way). The starred columns are
    the covariance the merge will use; the scale is the part that is independent
    between the catalogs of a group. Parameters a catalog does not measure print as
    ``nan``.
    """
    from .model import model_shared

    summary = catalog_summary(model, stars, keys, chunk)
    shared = model_shared(model)
    lines = [f"{'catalog':24s}{'n':>9s}   median scale: "
             + " ".join(f"{n:>7s}" for n in PARAM_NAMES)
             + "    median add*: "
             + " ".join(f"{n:>7s}" for n in PARAM_NAMES)
             + f"{'nu':>7s}{'median dln det/p*':>17s}",
             "-" * 111]
    for k, key in enumerate(summary["keys"]):
        label = registry.get_spec(key).label if key in registry.keys() else key
        sc, ad = summary["scale"][k], summary["add"][k]
        if not np.isfinite(np.r_[sc, ad]).any():
            continue  # a catalog that measures nothing has nothing to report
        # Per parameter: a catalog that measures three parameters has three numbers,
        # and a parameter it does not measure is ``nan``.
        nu_k = "-" if not np.isfinite(summary["nu"][k]) else f"{summary['nu'][k]:.1f}"
        lines.append(f"{label[:24]:24s}{int(summary['n'][k]):9,d}   "
                     + " ".join(f"{v:7.2f}" for v in sc) + "    "
                     + " ".join(f"{v:7.4f}" for v in ad)
                     + f"{nu_k:>7s}{summary['det'][k]:17.2f}")
    if shared is not None and np.any(np.asarray(shared) > 0):
        # Name the groups whose additive column carries a systematic, so the starred
        # numbers are not mistaken for a per-catalog property.
        known = all(key in registry.keys() for key in keys)
        g_labels = registry.group_labels(keys) if known else tuple(keys)
        for gi in np.flatnonzero(np.max(np.asarray(shared), axis=1) > 0):
            tau = np.sqrt(np.maximum(np.asarray(shared)[gi], 0.0))
            lines.append("")
            lines.append(f"* {g_labels[gi]}: shared sigma "
                         + "/".join(f"{t:.4f}" for t in tau)
                         + " (" + ", ".join(PARAM_NAMES) + "), part of the starred columns")
    return "\n".join(lines)


def scale_add_table(models: Dict[str, object], stars, labels: Sequence[str],
                    max_entries: int = 400_000) -> str:
    """Per catalog and parameter: the fitted sigma scale and the additive sigma.

    Medians over the validation entries of each catalog, so the network is described
    by the same two numbers as the constant arms. ``sigma ratio`` is the combined
    effect at that catalog's median quoted error, ``sqrt(scale + (add + tau^2)/C)``
    -- the inflation of the error the merge actually applies, so it includes the
    group's shared variance ``tau^2`` and is the number to compare with 1. ``add sig``
    likewise shows the total additive sigma, ``sqrt(add + tau^2)``; for the members of
    a group both rows therefore carry the same shared part.
    """
    from .covariance import entry_cov_np, sigma_out_np
    from .model import model_shared, numpy_delta_fn

    names = ("x", "ls", "sig", "sig0", "J", "c", "g", "m")
    pick = {name: stars[name] for name in names}
    if len(pick["x"]) > max_entries:  # a stable subsample is enough for medians
        step = int(np.ceil(len(pick["x"]) / max_entries))
        pick = {name: v[::step] for name, v in pick.items()}
    x, ls, c, g, m = pick["x"], pick["ls"], pick["c"], pick["g"], pick["m"]
    ref_sig = sigma_out_np(entry_cov_np(pick["sig0"], pick["J"], measured=m))
    rows = []
    for name, fn in models.items():
        if not hasattr(fn, "trainable_variables"):
            continue
        scale, add = numpy_delta_fn(fn)(x, ls, g)
        s2 = np.exp(scale.astype("f8"))
        shared = model_shared(fn) if hasattr(fn, "shared_variance") else None
        total = entry_cov_np(pick["sig"], pick["J"], scale, add, measured=m)
        ii = np.arange(N_PARAMS)
        if shared is not None:
            total[:, ii, ii] += shared[g]
            add = add + np.log(shared[g] + 1e-30)  # additive variance, shared included
        a = np.exp(0.5 * add.astype("f8"))
        ratio = sigma_out_np(total) / ref_sig
        for k, lab in enumerate(labels):
            for p, pname in enumerate(PARAM_NAMES):
                sel = (c == k) & m[:, p]
                if sel.sum() < 100:
                    continue
                rows.append((name, lab, pname, float(np.sqrt(s2[sel, p].mean())),
                             float(np.sqrt((a[sel, p] ** 2).mean())),
                             float(np.median(ratio[sel, p]))))
    head = f"{'model':22s}{'catalog':24s}{'param':7s}{'scale(sig)':>11s}{'add sig':>9s}{'sig ratio':>10s}"
    lines = [head, "-" * len(head)]
    for name, lab, pname, s, a, r in rows:
        lines.append(f"{name:22s}{lab[:24]:24s}{pname:7s}{s:11.2f}{a:9.4f}{r:10.2f}")
    return "\n".join(lines)


def _fixed_np(fn):
    """A numpy delta function for the ``fixed_delta`` baselines."""

    def wrap(x, ls, c):
        import tensorflow as tf

        a, b = fn(tf.constant(x), tf.constant(ls), tf.constant(c, tf.int32))
        return a.numpy(), b.numpy()

    return wrap


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-delta-train",
        description="Learn extra uncertainties for the harmonized parameters (post hoc).",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the star table")
    parser.add_argument("--min-entries", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="do not use the GPU")
    for f in dataclasses.fields(DeltaConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):  # --flag / --no-flag, not bool("False")
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import tensorflow as tf

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    else:
        from ..model.io import enable_memory_growth

        enable_memory_growth()
    cfg = DeltaConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(DeltaConfig)})
    res = run(args.run_dir, args.out_dir, cfg, args.rebuild, args.min_entries)
    print(res["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
