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

With no options this writes, for each plane, three videos to
``<run-dir>/videos/``:

    kiel.mp4                    no highlighting
    kiel_control_10.mp4         SDSS-V (ASPCAP) highlighted
    kiel_pair_3_10.mp4          the (GALAH DR4, SDSS-V ASPCAP) pairs highlighted

and the same with ``feh_alpha`` in place of ``kiel``. Use ``--highlight KEY``,
``--pair KEY KEY`` (catalog keys; see ``astrotune-prepare --list``) and
``--plane`` for others, e.g. for the 1-parameter-vs-1-parameter videos

    python -m astrotune.videos --run-dir models/my_run --no-plain \\
        --plane param1v1_diff --pair galah_dr4 sdssv_aspcap
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from . import registry
from .training_data import load_training_data
from .training_plots import PARAM_LABELS, catalog_colors

log = logging.getLogger(__name__)

DEFAULT_HIGHLIGHT = "sdssv_aspcap"
DEFAULT_PAIR = ("galah_dr4", "sdssv_aspcap")

#: Values at or above this mark a parameter that a catalog does not measure
MISSING = 1e6

#: Fewest stars for which a video is worth making
MIN_STARS = 20


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


# --- Trajectories ---------------------------------------------------------------
def catalog_trajectories(bij, ctrl_dict, n_frames: int, n_plot: int) -> List[Trajectories]:
    """Trajectories of up to ``n_plot`` stars per catalog.

    ``ctrl_dict`` must be loaded with ``fix_unknown_errors=False``, so that
    ``xe_control >= 1e6`` identifies parameters a catalog does not measure.
    """
    import tensorflow as tf

    out = []
    for k in tqdm(range(int(ctrl_dict["c_control"].max()) + 1), desc="catalog paths"):
        idx = np.where((ctrl_dict["c_control"] == k) & ctrl_dict["from_catalog"])[0][:n_plot]
        x = ctrl_dict["x_control"][idx]
        valid = ctrl_dict["xe_control"][idx] < MISSING
        c = tf.fill([len(idx)], tf.constant(k, dtype=tf.int32))
        xt = bij.return_path(tf.constant(x, dtype=tf.float32), c, n_frames).numpy()
        out.append(Trajectories(xt, valid))
    return out


def paired_trajectories(
    bij, pair_dict, i0: int, i1: int, n_frames: int, n_max: int
) -> Tuple[Trajectories, Trajectories]:
    """Trajectories of the stars paired between catalogs ``i0`` and ``i1``.

    Returns the same ``n`` stars as measured by catalog ``i0`` and by catalog
    ``i1``. Pairs are stored with a random orientation, so both orientations
    are collected (``n_max // 2`` from each). A parameter counts as measured
    only if both catalogs measure it.
    """
    import tensorflow as tf

    def one_orientation(a, b):
        idx = np.where((pair_dict["c0"] == a) & (pair_dict["c1"] == b))[0][: n_max // 2]
        if len(idx) == 0:
            return None
        ca = tf.fill([len(idx)], tf.constant(a, dtype=tf.int32))
        cb = tf.fill([len(idx)], tf.constant(b, dtype=tf.int32))
        xa = bij.return_path(tf.constant(pair_dict["x0"][idx]), ca, n_frames).numpy()
        xb = bij.return_path(tf.constant(pair_dict["x1"][idx]), cb, n_frames).numpy()
        valid = (pair_dict["xs0"][idx] < MISSING) & (pair_dict["xs1"][idx] < MISSING)
        return xa, xb, valid

    forward = one_orientation(i0, i1)
    reverse = one_orientation(i1, i0)  # here x0 is from catalog i1
    parts0, parts1, valid = [], [], []
    if forward is not None:
        parts0.append(forward[0])
        parts1.append(forward[1])
        valid.append(forward[2])
    if reverse is not None:
        parts0.append(reverse[1])
        parts1.append(reverse[0])
        valid.append(reverse[2])
    if not parts0:
        raise ValueError(f"No pairs between catalogs {i0} and {i1}.")
    valid = np.concatenate(valid, axis=0)
    return (
        Trajectories(np.concatenate(parts0, axis=1), valid),
        Trajectories(np.concatenate(parts1, axis=1), valid),
    )


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

    ``pair`` is ``((i0, i1), (traj0, traj1))`` from :func:`paired_trajectories`.
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

    ``pair`` is ``((i0, i1), (traj0, traj1))`` from :func:`paired_trajectories`.
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
def make_videos(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    highlights: Sequence[str] = (),
    pairs: Sequence[Tuple[str, str]] = (),
    planes: Sequence[str] = ("kiel", "feh_alpha"),
    plain: bool = True,
    n_video_frames: int = 90,
    n_plot: int = 2048,
    pairs_path: Optional[Path] = None,
    all_pairs_min: Optional[int] = None,
) -> List[Path]:
    """Make the requested videos for a trained run; return their paths.

    ``planes`` may contain the pair-only ``param1v1`` and ``param1v1_diff``.
    With ``all_pairs_min`` every catalog pair with at least that many stars in
    common is added to ``pairs`` (as the notebook did, with 50).
    """
    from .model import load_bijector

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "videos"
    labels = registry.labels()[: len(registry.run_keys(run_dir))]
    colors = catalog_colors(len(labels))

    bij = load_bijector(run_dir)
    pair_dict, ctrl_dict = load_training_data(pairs_path, fix_unknown_errors=False)
    catalogs = catalog_trajectories(bij, ctrl_dict, n_video_frames, n_plot)
    pair_idx = {
        tuple(sorted((registry.category_index(ka), registry.category_index(kb))))
        for ka, kb in pairs
    }
    if all_pairs_min is not None:
        n_cat = len(labels)
        lo = np.minimum(pair_dict["c0"], pair_dict["c1"]).astype("i8")
        hi = np.maximum(pair_dict["c0"], pair_dict["c1"]).astype("i8")
        counts = np.bincount(lo * n_cat + hi, minlength=n_cat * n_cat)
        pair_idx |= {divmod(int(g), n_cat) for g in np.where(counts >= all_pairs_min)[0]}
    pair_trajs = {}
    for i0, i1 in sorted(pair_idx):
        pair_trajs[(i0, i1)] = paired_trajectories(
            bij, pair_dict, i0, i1, n_video_frames, n_plot
        )

    paths = []
    kw = dict(n_video_frames=n_video_frames)
    for name in planes:
        if name in PARAM1V1:
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
    parser.add_argument("--pairs-path", type=Path, default=None)
    parser.add_argument(
        "--plane", action="append", choices=list(PLANES) + list(PARAM1V1), default=[],
        help="which plane(s) to draw (repeatable; default: kiel and feh_alpha). "
        "param1v1 and param1v1_diff are pair-only: they need --pair or --all-pairs-min",
    )
    parser.add_argument(
        "--highlight", action="append", metavar="KEY", default=[],
        help="catalog key to highlight (repeatable)",
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
    parser.add_argument("--n-frames", type=int, default=90)
    parser.add_argument("--n-plot", type=int, default=2048, help="stars per catalog")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    highlights, pairs = args.highlight, [tuple(p) for p in args.pair]
    if not highlights and not pairs and not args.no_plain and args.all_pairs_min is None:  # the default set
        highlights, pairs = [DEFAULT_HIGHLIGHT], [DEFAULT_PAIR]

    make_videos(
        args.run_dir, args.out_dir, highlights, pairs,
        planes=args.plane or list(PLANES),
        plain=not args.no_plain, n_video_frames=args.n_frames, n_plot=args.n_plot,
        pairs_path=args.pairs_path, all_pairs_min=args.all_pairs_min,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
