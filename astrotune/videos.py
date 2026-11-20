"""Videos of the transformation from catalog to harmonized space.

Each video shows stars from every catalog (gray) moving along their ODE
trajectories to the harmonized space, in one of two planes:

    kiel         5 ln(Teff/5000 K) vs logg
    feh_alpha    [Fe/H] vs [alpha/Fe]

Two more kinds of video compare the two sides of a catalog pair parameter by
parameter (a 2x2 grid, one panel per parameter, catalog j on the horizontal
axis, catalog k on the vertical axis; the planes ``param1v1`` and
``param1v1_diff``, the latter plotting ``k - j`` on the vertical axis). They
only exist for pairs (``--pair``, or ``--all-pairs-min N``). If the
transformation works, the points collapse onto the diagonal / onto zero.

Optionally, one or more catalogs are highlighted in color, or the two sides of
a catalog pair (the same stars, as measured by each catalog) are highlighted:
if the transformation works, the two sets of points converge. Stars whose
catalog does not measure one of the plotted parameters are not drawn.

    python -m astrotune.videos --run-dir models/my_run

With no options this writes, for each of the two planes, three videos to
``<run-dir>/videos/``: the unhighlighted one, one with
``DEFAULT_HIGHLIGHT`` highlighted and one with ``DEFAULT_PAIR`` highlighted.

**Which training file.** The run's ``config.json`` says which one it used: a
latent-y run records ``observations_path``, a pair-loss run ``pairs_path``, and
that is what is loaded (``--data-path`` overrides it). The control points come
from either file; the stars of a pair come from the observations, where both
sides of a pair are rows of the same star, or from the legacy pair table.

**One integration pass.** Every video needs the control points of every catalog
(they are the gray background, and a highlighted catalog is a subset of them)
plus the stars of each requested pair. Those points are collected first, the
pair points are deduplicated by ``(star, catalog)`` so that a star in several
pair videos is integrated once, and the whole set goes through the ODE in a few
large batches of ``--batch-size`` rather than one call per catalog and per pair.
The solve is vectorised over the batch, so this is worth a lot: on the
20-catalog data with every pair of at least 10,000 shared stars (210k points,
90 frames each) it takes 35 s at 4k points per call, 9 s at 16k and 3 s at 64k.
Rendering is the cost that matters: ``n_video_frames`` matplotlib frames per
video.

Examples::

    # every plane, every catalog highlighted, the pairs with >= 10k shared stars
    python -m astrotune.videos --run-dir models/my_run --all-planes --highlight-all \\
        --all-pairs-min 10000

    # two planes, three catalogs highlighted
    python -m astrotune.videos --run-dir models/my_run \\
        --plane kiel feh_alpha --highlight apogee_dr17 galah_dr4 sdssv_apogeenet

    # the 1-parameter-vs-1-parameter video of one pair
    python -m astrotune.videos --run-dir models/my_run --no-plain \\
        --plane param1v1_diff --pair galah_dr4 sdssv_aspcap
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from . import config, registry
from .training_data import load_observations, load_pairs
from .training_plots import PARAM_LABELS, catalog_colors

log = logging.getLogger(__name__)

DEFAULT_HIGHLIGHT = "sdssv_aspcap"
DEFAULT_PAIR = ("galah_dr4", "sdssv_aspcap")

#: Values at or above this mark a parameter that a catalog does not measure
MISSING = 1e6

#: Fewest stars for which a video is worth making
MIN_STARS = 20

#: ``--standard-overview`` draws these planes: the two parameter planes plus the
#: parameter-by-parameter comparison, which only exists for pairs.
STANDARD_OVERVIEW_PLANES = ("kiel", "feh_alpha", "param1v1")
#: ... this many catalog pairs, i.e. every pair with *more* than this many stars
#: in common ...
STANDARD_OVERVIEW_MIN_SHARED = 10_000
#: ... plus each catalog's strongest few pairs, so that no catalog is missing
#: from the pair videos altogether.
STANDARD_OVERVIEW_TOP_K = 3

#: Pairs sampled per catalog pair for the pair videos. The shared-star *counts*
#: always come from the full overlap; this only limits how many stars are drawn.
PAIRS_PER_PAIR = 4096


@dataclass(frozen=True)
class Plane:
    name: str
    px: int  # parameter on the horizontal axis
    py: int  # parameter on the vertical axis

    @property
    def invert(self) -> Tuple[bool, bool]:
        # Teff and logg increase toward the lower left, as in a Kiel diagram
        return self.px in (0, 2), self.py in (0, 2)


PLANES = {
    "kiel": Plane("kiel", 0, 2),
    "feh_alpha": Plane("feh_alpha", 1, 3),
}

#: Pair-only videos: one parameter of catalog j against the same one of catalog k
PARAM1V1 = ("param1v1", "param1v1_diff")

#: Every plane, in the order the videos are made
ALL_PLANES = tuple(PLANES) + PARAM1V1

PARAM1V1_LABELS = (
    r"$T_\mathrm{eff} / 1000\,\mathrm{K}$",
    r"$\left[\mathrm{Fe/H}\right]$",
    r"$\log g$",
    r"$\left[\alpha/\mathrm{Fe}\right]$",
)


@dataclass
class Trajectories:
    """Trajectories ``xt`` of shape ``(n_frames, n, 4)`` and which parameters
    are measured for each star (``valid``, ``(n, 4)``)."""

    xt: np.ndarray
    valid: np.ndarray

    def select(self, plane: Plane) -> np.ndarray:
        """Trajectories of the stars with both of the plane's parameters measured."""
        m = self.valid[:, plane.px] & self.valid[:, plane.py]
        return self.xt[:, m]


def ffmpeg_exe() -> str:
    """Path to ffmpeg: one on ``PATH``, else the static binary of imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg not found; install it or `pip install imageio-ffmpeg`."
        ) from e


# --- The training data a run used ------------------------------------------------
@dataclass
class Source:
    """The training file a run used, and what the videos need from it."""

    kind: str  # "observations" or "pairs"
    path: Path
    ctrl: Dict[str, np.ndarray]  # x_control, xe_control, c_control, from_catalog
    n_cat: int
    keys: Tuple[str, ...]
    obs: Optional[Dict[str, np.ndarray]] = None
    pairs: Optional[Dict[str, np.ndarray]] = None
    _rows: Optional[np.ndarray] = None  # one row per (star, catalog), cached
    _stars: Optional[List[np.ndarray]] = None  # sorted stars per catalog, cached

    @property
    def description(self) -> str:
        return f"{self.path.name} ({self.kind}, {self.n_cat} catalogs)"

    def rows(self) -> np.ndarray:
        """Rows of ``obs``: the first observation of every (star, catalog)."""
        if self._rows is None:
            key = self.obs["star"].astype("i8") * self.n_cat + self.obs["c"]
            self._rows = np.unique(key, return_index=True)[1]
        return self._rows

    def stars(self, k: int) -> np.ndarray:
        """Sorted stars catalog ``k`` observes."""
        if self._stars is None:
            rows = self.rows()
            star, c = self.obs["star"][rows], self.obs["c"][rows]
            self._stars = [np.sort(star[c == j]) for j in range(self.n_cat)]
        return self._stars[k]

    def shared_count(self, i0: int, i1: int) -> int:
        """How many stars both catalogs observe (pair rows, for the legacy table)."""
        if self.kind == "observations":
            a, b = self.stars(i0), self.stars(i1)
            return int(np.intersect1d(a, b, assume_unique=True).size)
        c0, c1 = self.pairs["c0"], self.pairs["c1"]
        return int((((c0 == i0) & (c1 == i1)) | ((c0 == i1) & (c1 == i0))).sum())


def resolve_data_path(path) -> Path:
    """A path recorded in ``config.json``: relative ones are from the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def load_source(run_dir, path=None) -> Source:
    """Load the training data the run used, as named by its ``config.json``.

    A latent-y run records ``observations_path`` and a pair-loss run
    ``pairs_path``; ``path`` overrides either. Runs from before those keys were
    written fall back to whichever file exists.
    """
    run_dir = Path(run_dir)
    try:
        with open(run_dir / "config.json") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    keys = registry.run_keys(run_dir)

    kind, chosen = None, None
    if path is not None:
        chosen = resolve_data_path(path)
        kind = "observations" if "observation" in chosen.name else "pairs"
    elif cfg.get("observations_path"):
        kind, chosen = "observations", resolve_data_path(cfg["observations_path"])
    elif cfg.get("pairs_path"):
        kind, chosen = "pairs", resolve_data_path(cfg["pairs_path"])
    if chosen is None:
        raise FileNotFoundError(
            f"cannot tell which training file {run_dir} used: its config.json has neither "
            f"`observations_path` nor `pairs_path`. Pass --data-path."
        )
    if not chosen.exists():
        raise FileNotFoundError(f"{chosen} does not exist (from {run_dir}/config.json); pass --data-path")

    if kind == "observations":
        obs, ctrl = load_observations(chosen, fix_unknown_errors=False)
        n_cat = int(ctrl["c_control"].max()) + 1
        return Source(kind, chosen, ctrl, n_cat, keys, obs=obs)
    pairs, ctrl = load_pairs(chosen, fix_unknown_errors=False)
    n_cat = max(int(ctrl["c_control"].max()), int(pairs["c0"].max()), int(pairs["c1"].max())) + 1
    return Source(kind, chosen, ctrl, n_cat, keys, pairs=pairs)


# --- What has to be integrated ---------------------------------------------------
@dataclass
class Plan:
    """The observations every requested video needs, before any integration.

    ``x``, ``c`` and ``valid`` are the points to integrate; ``ctrl_idx`` and
    ``pair_idx`` say which of them each video uses.
    """

    x: np.ndarray
    c: np.ndarray
    valid: np.ndarray
    ctrl_idx: List[np.ndarray]
    pair_idx: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    n_ctrl: int = 0
    n_pair: int = 0
    n_pair_before_dedup: int = 0

    def __len__(self) -> int:
        return len(self.c)


def _pair_rows(source: Source, i0: int, i1: int, n_max: int, rng) -> Optional[dict]:
    """Both sides of up to ``n_max`` stars observed by catalogs ``i0`` and ``i1``.

    Returns the catalog coordinates, the mask of parameters both catalogs measure
    and a dedupe key ``(star, catalog)`` per row, or None if the pair is empty.
    """
    if source.kind == "observations":
        obs = source.obs
        rows = source.rows()
        star, c = obs["star"][rows], obs["c"][rows]
        in0, in1 = np.flatnonzero(c == i0), np.flatnonzero(c == i1)
        common, j0, j1 = np.intersect1d(star[in0], star[in1], assume_unique=True, return_indices=True)
        if len(common) == 0:
            return None
        if len(common) > n_max:  # a random sample, not the first n by id (that is a patch of sky)
            sel = np.sort(rng.permutation(len(common))[:n_max])
            j0, j1 = j0[sel], j1[sel]
        r0, r1 = rows[in0[j0]], rows[in1[j1]]
        xs0, xs1 = obs["xs"][r0], obs["xs"][r1]
        return dict(
            x0=obs["x"][r0], x1=obs["x"][r1],
            valid=(xs0 < MISSING) & (xs1 < MISSING),
            key0=np.stack([obs["id"][r0], np.full(len(r0), i0)], axis=1),
            key1=np.stack([obs["id"][r1], np.full(len(r1), i1)], axis=1),
        )
    pairs = source.pairs
    forward = np.flatnonzero((pairs["c0"] == i0) & (pairs["c1"] == i1))[: n_max // 2]
    reverse = np.flatnonzero((pairs["c0"] == i1) & (pairs["c1"] == i0))[: n_max // 2]
    if len(forward) == 0 and len(reverse) == 0:
        return None
    x0 = np.concatenate([pairs["x0"][forward], pairs["x1"][reverse]])
    x1 = np.concatenate([pairs["x1"][forward], pairs["x0"][reverse]])
    xs0 = np.concatenate([pairs["xs0"][forward], pairs["xs1"][reverse]])
    xs1 = np.concatenate([pairs["xs1"][forward], pairs["xs0"][reverse]])
    # The legacy table has no star ids, so its pairs can share nothing; a key unique
    # per row keeps the code path uniform without merging anything.
    key = np.arange(len(x0), dtype="i8") + (1 << 40)
    return dict(x0=x0, x1=x1, valid=(xs0 < MISSING) & (xs1 < MISSING),
                key0=np.stack([key, np.full(len(key), i0)], axis=1),
                key1=np.stack([key, np.full(len(key), i1)], axis=1))


def build_plan(source: Source, pair_list: Sequence[Tuple[int, int]], n_plot: int,
               rng=None) -> Plan:
    """Collect the points every video needs, and dedupe the pair points by star."""
    rng = rng if rng is not None else np.random.default_rng(0)
    ctrl = source.ctrl

    # Control points: the gray background of every plane video, and the highlighted
    # subset of the highlight videos (so those need nothing extra).
    xs: List[np.ndarray] = []
    cs: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    ctrl_idx: List[np.ndarray] = []
    for k in range(source.n_cat):
        idx = np.flatnonzero((ctrl["c_control"] == k) & ctrl["from_catalog"])[:n_plot]
        start = sum(len(a) for a in xs)
        ctrl_idx.append(np.arange(start, start + len(idx)))
        xs.append(ctrl["x_control"][idx])
        cs.append(np.full(len(idx), k, dtype="i4"))
        vs.append(ctrl["xe_control"][idx] < MISSING)
    n_ctrl = sum(len(a) for a in xs)

    px: List[np.ndarray] = []
    pc: List[np.ndarray] = []
    pk: List[np.ndarray] = []
    pair_idx: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (i0, i1) in pair_list:
        got = _pair_rows(source, i0, i1, n_plot, rng)
        if got is None:
            log.warning("No stars observed by both %s and %s; skipping their videos",
                        source.keys[i0], source.keys[i1])
            continue
        n = len(got["x0"])
        off = sum(len(a) for a in px)
        pair_idx[(i0, i1)] = (np.arange(off, off + n), np.arange(off + n, off + 2 * n), got["valid"])
        px += [got["x0"], got["x1"]]
        pc += [np.full(n, i0, "i4"), np.full(n, i1, "i4")]
        pk += [got["key0"], got["key1"]]

    n_pair_raw = sum(len(a) for a in px)
    if px:
        x_pair, c_pair = np.concatenate(px), np.concatenate(pc)
        _, first, inverse = np.unique(np.concatenate(pk, axis=0), axis=0,
                                      return_index=True, return_inverse=True)
        # Number the distinct points by first use, so a star that two pairs share is
        # integrated once and both videos point at the same path.
        order = np.argsort(first)
        rank = np.empty(len(first), dtype="i8")
        rank[order] = np.arange(len(first))
        point_of_row = rank[inverse] + n_ctrl
        pair_idx = {k: (point_of_row[a], point_of_row[b], v) for k, (a, b, v) in pair_idx.items()}
        xs.append(x_pair[first[order]])
        cs.append(c_pair[first[order]])
        vs.append(np.ones((len(first), config.N_PARAMS), dtype=bool))
        n_pair = len(first)
    else:
        n_pair = 0

    return Plan(
        x=np.concatenate(xs).astype("f4"),
        c=np.concatenate(cs).astype("i4"),
        valid=np.concatenate(vs),
        ctrl_idx=ctrl_idx,
        pair_idx=pair_idx,
        n_ctrl=n_ctrl,
        n_pair=n_pair,
        n_pair_before_dedup=n_pair_raw,
    )


def integrate(bij, plan: Plan, n_frames: int, batch_size: int = 16384) -> np.ndarray:
    """Integrate every point of the plan; returns paths ``(n_frames, n, 4)``.

    One call per ``batch_size`` points, not one per catalog and per pair: the ODE
    solve is vectorised over the batch, so a few large calls do the work of many
    small ones.
    """
    import tensorflow as tf

    n = len(plan)
    out = np.empty((n_frames, n, 4), dtype="f4")
    n_batches = 0
    for start in tqdm(range(0, n, batch_size), desc="integrating"):
        sl = slice(start, min(start + batch_size, n))
        xt = bij.return_path(
            tf.constant(plan.x[sl], dtype=tf.float32),
            tf.constant(plan.c[sl], dtype=tf.int32),
            n_frames,
        ).numpy()
        out[:, sl] = xt
        n_batches += 1
    log.info("Integrated %d observations in %d batch(es) of up to %d", n, n_batches, batch_size)
    return out


def make_trajectories(plan: Plan, xt: np.ndarray) -> Tuple[List[Trajectories], Dict]:
    """Cut the integrated paths into the per-catalog and per-pair trajectories."""
    catalogs = [Trajectories(xt[:, idx], plan.valid[idx]) for idx in plan.ctrl_idx]
    pairs = {
        key: (Trajectories(xt[:, a], valid), Trajectories(xt[:, b], valid))
        for key, (a, b, valid) in plan.pair_idx.items()
    }
    return catalogs, pairs


# --- Frames ---------------------------------------------------------------------
def interpolate_at_time(arr: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate trajectories ``(n_frames, n, d)`` at time ``t`` in [0, 1]."""
    n_frames = arr.shape[0]
    t_idx = t * (n_frames - 1)
    i0 = int(np.floor(t_idx))
    i1 = min(i0 + 1, n_frames - 1)
    alpha = t_idx - i0
    return (1 - alpha) * arr[i0] + alpha * arr[i1]


def get_frame_times(n_video_frames: int, ease: str) -> np.ndarray:
    if ease == "linear":
        return np.linspace(0.0, 1.0, n_video_frames)
    if ease == "sigmoid":
        t = 0.5 * (np.tanh(np.linspace(-2, 2, n_video_frames)) + 1)
        t -= t[0]
        return t / t[-1]
    if ease == "sine":
        return 0.5 * (1 + np.sin(np.pi * np.linspace(-0.5, 0.5, n_video_frames)))
    raise ValueError('"ease" must be "linear", "sigmoid" or "sine".')


def plane_limits(starts: np.ndarray, plane: Plane):
    """Axis limits from the catalogs' original coordinates, with margins.

    ``starts`` is ``(n, 4)``: the stars at time 0.
    """
    lim = np.percentile(starts[:, [plane.px, plane.py]], [1.0, 99.0], axis=0)
    w = lim[1] - lim[0]
    lim[0] -= 0.3 * w
    lim[1] += 0.3 * w
    xlim, ylim = lim.T
    inv_x, inv_y = plane.invert
    return (xlim[::-1] if inv_x else xlim), (ylim[::-1] if inv_y else ylim)


def format_teff_axis(axis, lim: Tuple[float, float]) -> None:
    """Label an axis holding ``5 ln(Teff / 5000 K)`` with Teff in units of 1000 K.

    ``axis`` is ``ax.xaxis`` or ``ax.yaxis`` and ``lim`` its (fixed) limits. The
    tick spacing is chosen from the limits, so that a narrow range (say 3-4
    thousand K) is still labeled.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    t_lo, t_hi = sorted(5.0 * np.exp(np.asarray(lim, dtype="f8") / 5.0))  # 1000 K
    step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10) if (t_hi - t_lo) / s <= 6), 10)
    ticks = np.arange(np.ceil(t_lo / step), np.floor(t_hi / step) + 1) * step
    axis.set_major_locator(FixedLocator(5 * np.log(ticks / 5)))
    axis.set_minor_locator(NullLocator())
    fmt = "{:.1f}" if step < 1 else "{:.0f}"
    axis.set_major_formatter(FuncFormatter(lambda x, pos: fmt.format(5.0 * np.exp(x / 5.0))))


def param1v1_limits(a: np.ndarray, b: np.ndarray, plot_diff: bool):
    """Axis limits ``(xlim, ylim)`` for one panel, from all frames.

    ``a`` and ``b`` are the ``(n_frames, n)`` values of the two catalogs.
    Percentiles (0.5-99.5) plus a margin, so a few outliers do not set the scale.
    """

    def span(v):
        lo, hi = np.percentile(v, [0.5, 99.5])
        w = max(hi - lo, 1e-3)
        return lo - 0.1 * w, hi + 0.1 * w

    if plot_diff:
        xlim = span(a)
        ylo, yhi = span(b - a)
        return xlim, (min(ylo, 0.0), max(yhi, 0.0))
    lim = span(np.concatenate([a.ravel(), b.ravel()]))
    return lim, lim


def make_param1v1_video(
    out_path: Path,
    pair: Tuple[Tuple[int, int], Tuple[Trajectories, Trajectories]],
    labels: Sequence[str],
    plot_diff: bool = False,
    n_video_frames: int = 90,
    ease: str = "sine",
) -> Path:
    """Parameter-by-parameter comparison of the stars paired between two catalogs.

    ``pair`` is ``((i0, i1), (traj0, traj1))`` from :func:`make_trajectories`.
    Each panel shows one parameter of catalog ``i0`` (horizontal) against the
    same parameter of ``i1`` (vertical), or against ``i1 - i0`` if ``plot_diff``,
    for the stars for which both catalogs measure it.
    """
    from matplotlib.ticker import AutoMinorLocator

    (i0, i1), (t0, t1) = pair
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="tight", dpi=100)

    panels = []  # (scatter, a, b) per drawn panel
    for p, ax in enumerate(axes.flat):
        ax.set_xlabel(PARAM1V1_LABELS[p])
        m = t0.valid[:, p]
        if m.sum() < MIN_STARS:
            ax.text(0.5, 0.5, "not measured by both catalogs", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        a, b = t0.xt[:, m, p], t1.xt[:, m, p]
        xlim, ylim = param1v1_limits(a, b, plot_diff)
        y0 = b[0] - a[0] if plot_diff else b[0]
        sc = ax.scatter(a[0], y0, s=2, alpha=0.2, edgecolors="none")
        if plot_diff:
            ax.axhline(0, c="k", alpha=0.3)
        else:
            ax.plot(xlim, xlim, c="k", alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if p == 0:  # Teff coordinate: label in 1000 K, increasing to the left
            format_teff_axis(ax.xaxis, xlim)
            ax.set_xlim(xlim[::-1])
            if not plot_diff:
                format_teff_axis(ax.yaxis, ylim)
                ax.set_ylim(ylim[::-1])
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, alpha=0.2)
        panels.append((sc, a, b))

    fig.supxlabel(labels[i0])
    fig.supylabel(f"{labels[i1]} - {labels[i0]}" if plot_diff else labels[i1])

    t_frames = get_frame_times(n_video_frames, ease)
    with tempfile.TemporaryDirectory(prefix="astrotune_frames_") as tmp:
        tmp = Path(tmp)
        for j, t in enumerate(tqdm(t_frames, desc=out_path.name)):
            for sc, a, b in panels:
                x, y = interpolate_at_time(a, t), interpolate_at_time(b, t)
                sc.set_offsets(np.c_[x, y - x if plot_diff else y])
            fig.savefig(tmp / f"frame_{j:03d}.png", dpi=200)
            if j == 0:
                fig.set_layout_engine(None)  # keep the layout fixed across frames
        encode_video(tmp, out_path, n_video_frames)
    plt.close(fig)
    return out_path


def encode_video(frames_dir: Path, out_path: Path, n_frames: int, crf: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "image2", "-framerate", "30",
        "-i", str(frames_dir / "frame_%03d.png"),
        "-frames:v", str(n_frames),
        "-profile:v", "main", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", str(crf),
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def make_plane_video(
    out_path: Path,
    plane: Plane,
    catalogs: Sequence[Trajectories],
    labels: Sequence[str],
    colors,
    c_highlight: Sequence[int] = (),
    pair: Optional[Tuple[Tuple[int, int], Tuple[Trajectories, Trajectories]]] = None,
    n_video_frames: int = 90,
    ease: str = "sigmoid",
    title: str = "",
) -> Path:
    """Render and encode one video.

    ``pair`` is ``((i0, i1), (traj0, traj1))`` from :func:`make_trajectories`.
    """
    kw = dict(edgecolors="none", s=4)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    xt_cat = [cat.select(plane) for cat in catalogs]
    xlim, ylim = plane_limits(np.concatenate([x[0] for x in xt_cat], axis=0), plane)

    scatter = []
    for k, (label, x) in enumerate(zip(labels, xt_cat)):
        on = k in c_highlight
        scatter.append(
            ax.scatter(
                x[0, :, plane.px], x[0, :, plane.py],
                color=colors[k] if on else "gray",
                alpha=0.5 if on else 0.2,
                zorder=100 if on else 0,
                label=label if on else None,
                **kw,
            )
        )
    xt_pair, scatter_p = [], []
    if pair is not None:
        (i0, i1), trajs = pair
        for cat, traj in zip((i0, i1), trajs):
            xt = traj.select(plane)
            xt_pair.append(xt)
            scatter_p.append(
                ax.scatter(
                    xt[0, :, plane.px], xt[0, :, plane.py],
                    color=colors[cat], alpha=0.4, zorder=110,
                    label=labels[cat] + " (pairs)", **kw,
                )
            )
    if c_highlight or pair is not None:
        ax.legend(loc="upper left", markerscale=4)

    ax.set_xlabel(PARAM_LABELS[plane.px])
    ax.set_ylabel(PARAM_LABELS[plane.py])
    ax.grid(True, alpha=0.2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=10)

    t_frames = get_frame_times(n_video_frames, ease)
    with tempfile.TemporaryDirectory(prefix="astrotune_frames_") as tmp:
        tmp = Path(tmp)
        for j, t in enumerate(tqdm(t_frames, desc=out_path.name)):
            for sc, xt in zip(scatter + scatter_p, xt_cat + xt_pair):
                x = interpolate_at_time(xt, t)
                sc.set_offsets(np.c_[x[:, plane.px], x[:, plane.py]])
            fig.savefig(tmp / f"frame_{j:03d}.png", dpi=200)
        encode_video(tmp, out_path, n_video_frames)
    plt.close(fig)
    return out_path


# --- Driver ---------------------------------------------------------------------
def _shared_counts(source: Source) -> np.ndarray:
    """Number of stars each catalog pair has in common, ``(n_cat, n_cat)``.

    The matrix ``--all-pairs-min`` and the standard overview select on. It is
    read from the source's own data (:meth:`Source.shared_count`), which for a
    run trained on observations is the full overlap of the two catalogs over
    every star in the file -- not the pairs of some earlier file, and not a
    sample: the pair videos draw their stars separately, this only decides which
    pairs are worth drawing.
    """
    n_cat = source.n_cat
    counts = np.zeros((n_cat, n_cat), dtype="i8")
    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            counts[i, j] = counts[j, i] = source.shared_count(i, j)
    return counts


def _overview_pairs(counts: np.ndarray) -> List[Tuple[int, int]]:
    """The pairs of the standard overview: above the threshold, plus the top few.

    Every pair with more than :data:`STANDARD_OVERVIEW_MIN_SHARED` stars in
    common, and on top of that each catalog's :data:`STANDARD_OVERVIEW_TOP_K`
    strongest connections, so that a catalog with few overlaps is still
    represented in the pair videos.
    """
    n_cat = counts.shape[0]
    chosen = {
        (int(i), int(j))
        for i, j in zip(*np.where(np.triu(counts, k=1) > STANDARD_OVERVIEW_MIN_SHARED))
    }

    for c in range(n_cat):
        partners = [
            (int(counts[min(c, d), max(c, d)]), d) for d in range(n_cat) if d != c
        ]
        partners = [(n, d) for n, d in partners if n > 0]
        partners.sort(key=lambda item: (-item[0], item[1]))
        for _, d in partners[:STANDARD_OVERVIEW_TOP_K]:
            chosen.add((min(c, d), max(c, d)))

    return sorted(chosen)


def make_videos(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    highlights: Sequence[str] = (),
    pairs: Sequence[Tuple[str, str]] = (),
    planes: Sequence[str] = ("kiel", "feh_alpha"),
    plain: bool = True,
    n_video_frames: int = 90,
    n_plot: int = 2048,
    data_path: Optional[Path] = None,
    all_pairs_min: Optional[int] = None,
    batch_size: int = 16384,
    standard_overview: bool = False,
) -> List[Path]:
    """Make the requested videos for a trained run; return their paths.

    ``planes`` may contain the pair-only ``param1v1`` and ``param1v1_diff``.
    With ``all_pairs_min`` every catalog pair with at least that many stars in
    common is added to ``pairs`` (as the notebook did, with 50).

    The data are the run's own: :func:`load_source` reads the training file the
    run's ``config.json`` names -- its observations, for any latent-y run -- and
    the pair counts, the pair stars and the control points all come from that
    one file. ``data_path`` overrides it.

    With ``standard_overview`` the run's usual set is added: the planes in
    :data:`STANDARD_OVERVIEW_PLANES`, the unhighlighted videos, one highlight
    per catalog, and the pairs of :func:`_overview_pairs` -- every pair with
    more than :data:`STANDARD_OVERVIEW_MIN_SHARED` shared stars plus each
    catalog's strongest :data:`STANDARD_OVERVIEW_TOP_K`.
    """
    from .model import load_bijector
    from .model.io import enable_memory_growth

    enable_memory_growth()
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "videos"
    source = load_source(run_dir, data_path)
    log.info("Training data: %s", source.description)
    labels = [registry.get_spec(k).label for k in source.keys]
    if len(labels) < source.n_cat:  # the file has catalogs the registry no longer lists
        labels = [registry.get_spec(k).label for k in registry.keys()[: source.n_cat]]
    colors = catalog_colors(len(labels))
    if standard_overview:
        # every catalog of the run highlighted individually, plus whatever was
        # asked for; the rest of the overview set is added once the pair counts
        # are known.
        highlights = list(dict.fromkeys([*registry.run_keys(run_dir), *highlights]))

    pair_idx = {
        tuple(sorted((registry.category_index(ka), registry.category_index(kb))))
        for ka, kb in pairs
    }
    if all_pairs_min is not None or standard_overview:
        counts = _shared_counts(source)
    if all_pairs_min is not None:
        rows, cols = np.where(counts >= all_pairs_min)
        pair_idx |= {(int(i), int(j)) for i, j in zip(rows, cols) if i < j}
    if standard_overview:
        overview = _overview_pairs(counts)
        log.info(
            "standard overview: %d catalog pairs (%d with more than %s shared stars)",
            len(overview),
            sum(counts[i, j] > STANDARD_OVERVIEW_MIN_SHARED for i, j in overview),
            f"{STANDARD_OVERVIEW_MIN_SHARED:,}",
        )
        pair_idx |= set(overview)

    missing = [p for p in pair_idx if max(p) >= source.n_cat]
    if missing:
        raise ValueError(
            f"Pairs {missing} name catalogs the run does not have ({source.n_cat})."
        )

    plan = build_plan(source, sorted(pair_idx), n_plot)
    if plan.n_pair_before_dedup:
        log.info(
            "Pair stars: %d before dedupe, %d distinct (star, catalog) rows",
            plan.n_pair_before_dedup, plan.n_pair,
        )
    log.info(
        "%d observations to integrate (%d control points + %d pair rows) in batches of %d",
        len(plan), plan.n_ctrl, plan.n_pair, batch_size,
    )
    xt = integrate(load_bijector(run_dir), plan, n_video_frames, batch_size)
    catalogs, pair_trajs = make_trajectories(plan, xt)
    del xt

    paths = []
    kw = dict(n_video_frames=n_video_frames)
    for name in planes:
        if name in PARAM1V1:
            if not pair_trajs:
                log.warning("Skipping %s: no catalog pairs requested", name)
                continue
            for (i0, i1), trajs in pair_trajs.items():
                n = int(trajs[0].valid.any(axis=1).sum())
                if n < MIN_STARS:
                    log.warning("Skipping %s pair (%s, %s): only %d paired stars",
                                name, labels[i0], labels[i1], n)
                    continue
                paths.append(
                    make_param1v1_video(
                        out_dir / f"{name}_pair_{i0}_{i1}.mp4", ((i0, i1), trajs), labels,
                        plot_diff=name.endswith("_diff"), **kw,
                    )
                )
            continue
        plane = PLANES[name]
        if plain:
            paths.append(
                make_plane_video(out_dir / f"{name}.mp4", plane, catalogs, labels, colors, **kw)
            )
        for key in highlights:
            c = registry.category_index(key)
            n = catalogs[c].select(plane).shape[1]
            if n < MIN_STARS:
                log.warning("Skipping %s highlight of %s: only %d stars with both parameters",
                            name, labels[c], n)
                continue
            paths.append(
                make_plane_video(out_dir / f"{name}_control_{c}.mp4", plane, catalogs, labels,
                                 colors, c_highlight=[c], title=f"{labels[c]} highlighted", **kw)
            )
        for (i0, i1), trajs in pair_trajs.items():
            n = trajs[0].select(plane).shape[1]
            if n < MIN_STARS:
                log.warning("Skipping %s pair (%s, %s): only %d paired stars with both parameters",
                            name, labels[i0], labels[i1], n)
                continue
            paths.append(
                make_plane_video(out_dir / f"{name}_pair_{i0}_{i1}.mp4", plane, catalogs, labels,
                                 colors, pair=((i0, i1), trajs),
                                 title=f"{labels[i0]} & {labels[i1]}: {n} paired stars", **kw)
            )
    for p in paths:
        log.info("Wrote %s", p)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-videos",
        description="Videos of the catalog -> harmonized transformation.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/videos")
    parser.add_argument(
        "--data-path", type=Path, default=None,
        help="training file to use; default: the one named in the run's config.json",
    )
    parser.add_argument(
        "--plane", nargs="+", choices=list(ALL_PLANES), default=None, metavar="PLANE",
        help=f"planes to draw, any of {', '.join(ALL_PLANES)} (default: kiel feh_alpha). "
        "param1v1 and param1v1_diff are pair-only: they need --pair or --all-pairs-min",
    )
    parser.add_argument(
        "--all-planes", action="store_true",
        help=f"every plane ({', '.join(ALL_PLANES)}); the pair-only ones need pairs",
    )
    parser.add_argument(
        "--highlight", nargs="+", metavar="KEY", default=None,
        help="catalog keys to highlight, one video each (default: %s)" % DEFAULT_HIGHLIGHT,
    )
    parser.add_argument(
        "--highlight-all", action="store_true", help="highlight every catalog of the run",
    )
    parser.add_argument(
        "--pair", action="append", nargs=2, metavar=("KEY0", "KEY1"), default=[],
        help="highlight the stars paired between two catalogs (repeatable)",
    )
    parser.add_argument(
        "--all-pairs-min", type=int, default=None, metavar="N",
        help="also use every catalog pair with at least N stars in common",
    )
    parser.add_argument("--no-plain", action="store_true", help="skip the unhighlighted video")
    parser.add_argument(
        "--standard-overview",
        action="store_true",
        help=(
            "also make the standard set: the kiel, feh_alpha and param1v1 planes, "
            "one highlight per catalog, and every pair with more than "
            f"{STANDARD_OVERVIEW_MIN_SHARED:,} shared stars plus each catalog's "
            f"top {STANDARD_OVERVIEW_TOP_K}"
        ),
    )
    parser.add_argument("--n-frames", type=int, default=90)
    parser.add_argument("--n-plot", type=int, default=2048, help="stars per catalog")
    parser.add_argument(
        "--batch-size", type=int, default=16384,
        help="observations integrated per ODE call",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")

    if args.plane and args.all_planes:
        parser.error("--plane and --all-planes are mutually exclusive")
    if args.highlight and args.highlight_all:
        parser.error("--highlight and --highlight-all are mutually exclusive")
    planes = list(ALL_PLANES) if args.all_planes else (args.plane or ["kiel", "feh_alpha"])
    if args.standard_overview:
        # the overview brings these; anything asked for explicitly is added to them
        planes = list(dict.fromkeys([*STANDARD_OVERVIEW_PLANES, *planes]))

    run_keys = registry.run_keys(args.run_dir)
    pairs = [tuple(p) for p in args.pair]
    if args.highlight_all:
        highlights = list(run_keys)
    elif args.highlight:
        highlights = list(args.highlight)
    elif (
        not args.no_plain
        and not args.standard_overview
        and not pairs
        and args.all_pairs_min is None
    ):  # the default set
        highlights = [k for k in [DEFAULT_HIGHLIGHT] if k in run_keys]
        pairs = [DEFAULT_PAIR] if set(DEFAULT_PAIR) <= set(run_keys) else []
        if not highlights and not pairs:
            log.warning(
                "This run has neither %s nor %s, so nothing is highlighted by "
                "default; pass --highlight or --highlight-all",
                DEFAULT_HIGHLIGHT, DEFAULT_PAIR,
            )
    else:
        highlights = []
    unknown = [k for k in highlights + [k for p in pairs for k in p] if k not in run_keys]
    if unknown:
        parser.error(f"unknown catalog key(s) {unknown}; the run has {list(run_keys)}")

    make_videos(
        args.run_dir, args.out_dir, highlights, pairs,
        planes=planes, plain=not args.no_plain, n_video_frames=args.n_frames, n_plot=args.n_plot,
        data_path=args.data_path, all_pairs_min=args.all_pairs_min,
        batch_size=args.batch_size, standard_overview=args.standard_overview,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
