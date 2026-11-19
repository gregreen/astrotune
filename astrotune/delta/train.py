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
            prior = cfg.prior_weight * prior_penalty(batch, params)
            loss = data + tf.cast(prior, tf.float64) + tf.cast(cfg.l2_weights * l2_penalty(model), tf.float64)
            if getattr(model, "tail", None) is not None:
                # One Gaussian prior per catalog on the unconstrained dof coordinate
                loss += tf.cast(cfg.nu_prior_weight * model.tail.prior_penalty(
                    cfg.nu_prior_mean, cfg.nu_prior_sd), tf.float64)
            if getattr(model, "shared", None) is not None:
                # ... and one that pulls each group's shared variance down to the floor
                loss += tf.cast(cfg.shared_prior_weight * model.shared.prior_penalty(),
                                tf.float64)
            if control is not None and cfg.control_weight:
                # Away from the data the model is pulled to scale 1, add 0: the same
                # prior, evaluated on points the likelihood never sees.
                loss += tf.cast(cfg.control_weight * control_penalty(control, model), tf.float64)
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, cfg.clip_norm)
        opt.apply_gradients(zip(grads, variables))
        return data

    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n_star)
        losses = []
        for idx, seg, n in tqdm(block_indices(stars, perm, cfg.batch_size),
                                total=steps, desc=f"{desc} {epoch + 1}/{epochs}", leave=False):
            losses.append(float(step(make_batch(stars, idx, seg))))
        rec = dict(epoch=epoch + 1, train_nll=float(np.mean(losses)))
        if val is not None:
            rec["val_nll"] = mean_star_nll(val, model, max_stars=cfg.val_max_stars)
        history.append(rec)
        log.info("%s epoch %d/%d: %s", desc, epoch + 1, epochs,
                 ", ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "epoch"))
    return history


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


def per_catalog_table(model, stars, keys: Sequence[str], chunk: int = 65_536) -> str:
    """One row per catalog: size, the two learned terms, the tail weight, and the effect.

    The rows are the *validation* entries of each catalog (the sample every other
    number in ``evaluation/`` is measured on). ``median scale`` is the median of the
    sigma scale ``exp(ln_scale / 2)`` over those entries and the parameters they
    measure (1 = the quoted error is left alone), ``median add*`` the same for the
    additive sigma *including the group's shared variance* (the floor is 0.003, so a
    value above it is the systematic the catalog shares with the rest of its group),
    ``nu`` the group's degrees of freedom, and ``median dln det*`` the median change in
    the log determinant of the total entry covariance ``S + diag(tau^2)``,
    ``ln det - ln det C_m`` over the measured submatrix -- how much volume the
    correction adds or removes, and the one number that combines all three terms
    (four parameters at scale 0.35 with nothing additive would be ``8 ln 0.35 =
    -8.4``). The starred columns are the covariance the merge will use; the scale is
    the part that is independent between the catalogs of a group.
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
                - np.linalg.slogdet(old[rows][:, mask][:, :, mask])[1])

    lines = [f"{'catalog':24s}{'n':>9s}{'median scale':>14s}{'median add*':>13s}"
             f"{'nu':>7s}{'median dln det*':>17s}",
             "-" * 83]
    for k, key in enumerate(keys):
        label = registry.get_spec(key).label if key in registry.keys() else key
        in_cat = stars["c"] == k
        measured = in_cat[:, None] & stars["m"]
        if not measured.any():
            continue
        sc = float(np.median(s_val[measured]))
        ad = float(np.median(a_val[measured]))
        de = dlndet[in_cat & np.isfinite(dlndet)]
        nu_k = "-" if nu is None else f"{nu[stars['g'][np.argmax(in_cat)]]:.1f}"
        lines.append(f"{label[:24]:24s}{int(in_cat.sum()):9,d}{sc:14.2f}{ad:12.4f}"
                     f"{nu_k:>7s}{np.median(de) if de.size else np.nan:16.2f}")
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
