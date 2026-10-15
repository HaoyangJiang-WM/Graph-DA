from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image

DEFAULT_METHODS = (
    ("noda", "NoDA"),
    ("etkf", "Traditional DA · ETKF"),
    ("graph_nao", "Graph-NAO · ours"),
)


def _energy(member_depth: np.ndarray, truth: np.ndarray, area: np.ndarray) -> float:
    """Empirical-distribution Energy Score in the registered depth metric."""

    scale = np.sqrt(area / area.sum())
    sample = member_depth * scale[None]
    target = truth * scale
    first = np.linalg.norm(sample - target[None], axis=1).mean()
    pair = np.linalg.norm(sample[:, None] - sample[None, :], axis=2).mean()
    return float(first - 0.5 * pair)


def _wet_brier(
    member_depth: np.ndarray,
    truth: np.ndarray,
    area: np.ndarray,
    threshold_m: float,
) -> float:
    probability = (member_depth > threshold_m).mean(0)
    observed = truth > threshold_m
    return float(np.sum((probability - observed) ** 2 * (area / area.sum())))


def _truth_boundary(wet: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    source, target = edge_index
    crossing = wet[source] != wet[target]
    nodes = np.concatenate((source[crossing], target[crossing]))
    return np.unique(nodes[wet[nodes]]) if nodes.size else np.flatnonzero(wet)


def _crop(xy: np.ndarray, active: np.ndarray) -> tuple[float, float, float, float]:
    point = xy[active]
    if point.shape[0] < 2:
        point = xy
    lower = point.min(0)
    upper = point.max(0)
    span = np.maximum(upper - lower, 1.0e-3)
    margin = 0.08 * span
    return (
        float(lower[0] - margin[0]),
        float(upper[0] + margin[0]),
        float(lower[1] - margin[1]),
        float(upper[1] + margin[1]),
    )


def _sar_scene_masks(
    sar_value: np.ndarray,
    acquisition_time_minutes: np.ndarray,
    frame_time_minutes: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return current-scene positives and earlier positive history."""

    values = np.asarray(sar_value, dtype=np.float64)
    acquisition = np.asarray(acquisition_time_minutes, dtype=np.float64)
    if values.ndim != 2 or acquisition.shape != (values.shape[0],):
        raise ValueError("SAR scenes and acquisition times are inconsistent")
    if np.any(np.diff(acquisition) <= 0.0):
        raise ValueError("SAR acquisition times must be strictly increasing")
    scene = int(np.searchsorted(acquisition, frame_time_minutes, side="right") - 1)
    scene = max(scene, 0)
    current = values[scene] > 0.5
    history = (
        (values[:scene] > 0.5).any(0)
        if scene > 0
        else np.zeros_like(current)
    )
    return scene, current, history


def _major_adjustment_regions(
    adjustment_mask: np.ndarray,
    xy: np.ndarray,
    area: np.ndarray,
    edge_index: np.ndarray,
    *,
    minimum_area_m2: float = 1.0e6,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Return connected, presentation-scale DA adjustment regions.

    Connectivity is inherited from the registered hydraulic graph.  The fixed
    1 km² display cutoff prevents isolated pixels from being presented as a
    coherent rapid adjustment; it does not affect assimilation or scoring.
    """

    active = np.asarray(adjustment_mask, dtype=bool)
    parent = np.arange(active.size, dtype=np.int64)
    rank = np.zeros(active.size, dtype=np.int8)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    for first, second in np.asarray(edge_index, dtype=np.int64).T:
        if active[first] and active[second]:
            union(int(first), int(second))

    components: dict[int, list[int]] = {}
    for node in np.flatnonzero(active):
        components.setdefault(find(int(node)), []).append(int(node))

    regions: list[tuple[np.ndarray, np.ndarray, float]] = []
    for nodes in components.values():
        index = np.asarray(nodes, dtype=np.int64)
        component_area = float(area[index].sum())
        if component_area < minimum_area_m2:
            continue
        regions.append((xy[index].min(0), xy[index].max(0), component_area))
    return sorted(regions, key=lambda item: item[2], reverse=True)


def _native_lattice_mask(
    xy: np.ndarray,
    mask: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Place an unstructured clipped-lattice field on its native raster indices.

    Interior floodplain cells lie on a regular Cartesian lattice while clipped
    boundary cells retain irregular centroids.  Rounding to the nearest native
    lattice index traces the cell-union boundary without smoothing or selecting
    connected components.
    """

    point = np.asarray(xy, dtype=np.float64)
    value = np.asarray(mask, dtype=bool)
    remainder = np.mod(point, spacing)
    offset = np.median(remainder, axis=0)
    index = np.rint((point - offset[None]) / spacing).astype(np.int64)
    lower = index.min(0)
    upper = index.max(0)
    shifted = index - lower[None]
    grid = np.full((upper[1] - lower[1] + 1, upper[0] - lower[0] + 1), np.nan)
    for (column, row), active in zip(shifted, value, strict=True):
        previous = grid[row, column]
        grid[row, column] = float(active) if np.isnan(previous) else max(previous, active)
    x_coordinate = offset[0] + np.arange(lower[0], upper[0] + 1) * spacing
    y_coordinate = offset[1] + np.arange(lower[1], upper[1] + 1) * spacing
    return x_coordinate, y_coordinate, np.ma.masked_invalid(grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preview-last",
        type=Path,
        default=None,
        help="Optional PNG path for the last rendered frame (visual QA).",
    )
    parser.add_argument(
        "--preview-acquisition",
        type=Path,
        default=None,
        help="Optional PNG path for the latest SAR-acquisition frame.",
    )
    parser.add_argument("--fps", type=float, default=1.25)
    parser.add_argument("--wet-threshold-m", type=float, default=1.0e-3)
    parser.add_argument(
        "--mode",
        choices=("depth", "depth-binned", "extent", "extent-change"),
        default="depth",
        help=(
            "depth: mean water depth; extent: ensemble wet probability; "
            "extent-change: persistent/new/receded inundation relative to t=0"
        ),
    )
    parser.add_argument(
        "--include-truth-panel",
        action="store_true",
        help="Prepend the held-out hydraulic truth as a fourth comparison panel.",
    )
    parser.add_argument(
        "--nonlearning-only",
        action="store_true",
        help="Render only NoDA and the registered traditional DA baseline.",
    )
    parser.add_argument(
        "--focus-floodplain",
        action="store_true",
        help=(
            "Crop to the union of truth and every method's active 2-D floodplain "
            "support instead of forcing the full 1-D river network into view."
        ),
    )
    parser.add_argument(
        "--hide-sar-history",
        action="store_true",
        help="Show only detections from the latest SAR acquisition.",
    )
    parser.add_argument(
        "--sar-at-acquisition-only",
        action="store_true",
        help="Draw SAR crosses only on frames where a new scene is acquired.",
    )
    parser.add_argument(
        "--selection-label",
        default=None,
        help="Optional auditable case-selection subtitle override.",
    )
    parser.add_argument(
        "--compact-subtitle",
        action="store_true",
        help="Use the selection label verbatim without the standard comparison suffix.",
    )
    parser.add_argument(
        "--show-initial-truth-boundary",
        action="store_true",
        help="Overlay the time-zero truth boundary as a dashed expansion reference.",
    )
    parser.add_argument(
        "--truth-boundary-only",
        action="store_true",
        help="Do not add the identical translucent truth-area fill below method fields.",
    )
    parser.add_argument(
        "--hide-truth-boundary",
        action="store_true",
        help="Omit the held-out truth boundary from the display and legend.",
    )
    parser.add_argument(
        "--binary-extent",
        action="store_true",
        help="For extent mode, show only the majority inundation mask in one colour.",
    )
    parser.add_argument(
        "--circle-da-adjustment",
        action="store_true",
        help=(
            "At SAR acquisition frames, circle connected ETKF-vs-NoDA mean-depth "
            "adjustments of at least 0.50 m and 1 km² on the DA panel."
        ),
    )
    parser.add_argument(
        "--outline-da-added-extent",
        action="store_true",
        help=(
            "At SAR acquisition frames, outline the inundation extent present in "
            "ETKF but absent from NoDA, using the displayed wet-depth threshold."
        ),
    )
    parser.add_argument(
        "--emphasize-small-da-added-cells",
        action="store_true",
        help=(
            "Keep the registered added-extent contour and additionally draw the "
            "individual added control-volume outlines when at most 32 cells change."
        ),
    )
    args = parser.parse_args()
    if args.binary_extent and args.mode != "extent":
        raise ValueError("binary extent is available only in extent mode")

    with np.load(args.bundle) as archive:
        data = {name: archive[name] for name in archive.files}
    schema = str(data["schema"])
    if schema not in {
        "v921_three_way_da_promotion_bundle_v1",
        "v947_two_scene_da_visualization_bundle_v1",
        "v1093_four_scene_da_visualization_bundle_v1",
    }:
        raise ValueError("unexpected visualization bundle schema")
    event = str(data["event"])
    selection_label = args.selection_label or str(
        data.get("selection_label", "Selected strict demonstration case")
    )
    if "traditional_mean_depth_m" in data:
        traditional_key = "traditional"
        methods = (
            ("noda", "NoDA"),
            (traditional_key, str(data.get("traditional_label", "Traditional DA"))),
            ("graph_nao", "Graph-NAO · ours"),
        )
    else:
        traditional_key = "etkf"
        methods = (
            ("noda", "NoDA"),
            ("etkf", "Traditional DA · ETKF"),
            ("graph_nao", str(data.get("graph_label", "Graph-NAO · ours"))),
        )
    if args.nonlearning_only:
        methods = tuple(
            method for method in methods if method[0] in {"noda", traditional_key}
        )

    xy = np.asarray(data["node_xy_normalized"], dtype=np.float64)
    area = np.asarray(data["node_area_m2"], dtype=np.float64)
    if "terrain_normalized" in data:
        elevation = np.asarray(data["terrain_normalized"], dtype=np.float64)
    else:
        flood_elevation = np.asarray(data["node_elevation_m"], dtype=np.float64)
        river_count_legacy = int(data["river_count"])
        elevation = np.concatenate(
            (
                np.full(
                    river_count_legacy,
                    float(np.quantile(flood_elevation, 0.05)),
                    dtype=np.float64,
                ),
                flood_elevation,
            )
        )
    edge = np.asarray(data["edge_index"], dtype=np.int64)
    truth = np.asarray(data["truth_depth_m"], dtype=np.float64)
    time_minutes = np.asarray(data["time_minutes"], dtype=np.float64)
    river_count = int(data["river_count"])
    wet_threshold = float(args.wet_threshold_m)
    frame_count, nodes = truth.shape
    if xy.shape != (nodes, 2) or area.shape != (nodes,):
        raise ValueError("visualization bundle geometry is inconsistent")

    method_mean = {
        short: np.asarray(data[f"{short}_mean_depth_m"], dtype=np.float64)
        for short, _ in methods
    }
    method_member = {
        short: np.asarray(data[f"{short}_member_depth_m"], dtype=np.float64)
        for short, _ in methods
    }
    method_probability = {
        short: (member > wet_threshold).mean(1)
        for short, member in method_member.items()
    }
    floodplain = np.arange(nodes) >= river_count
    active = (truth > wet_threshold).any(0)
    for value in method_probability.values():
        active |= (value >= 0.125).any(0)
    if args.focus_floodplain:
        active[:river_count] = False
    else:
        active[:river_count] = True
    xmin, xmax, ymin, ymax = _crop(xy, active)

    # The 1-D river state uses an equivalent storage measure, not literal mapped
    # floodplain depth.  It must not set the colour scale for the 2-D panels.
    truth_floodplain_depth = truth[:, floodplain]
    depth_pool = [truth_floodplain_depth[truth_floodplain_depth > wet_threshold]]
    depth_pool.extend(
        value[:, floodplain][value[:, floodplain] > wet_threshold]
        for value in method_mean.values()
    )
    positive = np.concatenate([value for value in depth_pool if value.size])
    vmax = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    vmax = max(vmax, 0.10)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)
    depth_cmap = ListedColormap(
        ("#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b")
    )
    depth_boundaries = np.asarray(
        (wet_threshold, 0.20, 0.50, 1.00, 2.00, max(vmax, 2.000001)),
        dtype=np.float64,
    )
    if np.any(np.diff(depth_boundaries) <= 0.0):
        raise ValueError("binned depth display requires a threshold below 0.20 m")
    depth_binned_norm = BoundaryNorm(depth_boundaries, depth_cmap.N)
    extent_cmap = ListedColormap(("#76b7dd", "#08519c"))
    extent_norm = BoundaryNorm((0.125, 0.5, 1.000001), extent_cmap.N)
    terrain_norm = plt.Normalize(
        vmin=float(np.quantile(elevation, 0.02)),
        vmax=float(np.quantile(elevation, 0.98)),
    )

    source, target = edge
    river_edge = (source < river_count) & (target < river_count)
    floodplain_edge = (source >= river_count) & (target >= river_count)
    floodplain_spacing = float(
        np.median(
            np.linalg.norm(
                xy[source[floodplain_edge]] - xy[target[floodplain_edge]],
                axis=1,
            )
        )
    )
    floodplain_triangulation = mtri.Triangulation(
        xy[floodplain, 0], xy[floodplain, 1]
    )
    initial_truth_floodplain = (
        truth[0, floodplain] > wet_threshold
    ).astype(np.float64)
    sar_xy = data.get("sar_pixel_xy_normalized")
    sar_value = np.asarray(data["sar_observation_value"], dtype=np.float64)
    if sar_value.ndim == 1:
        sar_acquisition_time = np.asarray([time_minutes[0]], dtype=np.float64)
        sar_value = sar_value[None]
    elif sar_value.ndim == 2:
        sar_acquisition_time = np.asarray(
            data["sar_acquisition_time_minutes"], dtype=np.float64
        )
        if sar_acquisition_time.shape != (sar_value.shape[0],):
            raise ValueError("SAR acquisition times do not match scene rows")
    else:
        raise ValueError("SAR observations must have shape [pixel] or [scene,pixel]")
    if np.any(np.diff(sar_acquisition_time) <= 0.0):
        raise ValueError("SAR acquisition times must be strictly increasing")

    output_frames: list[Image.Image] = []
    acquisition_frame_indices: list[int] = []
    panels = (
        (("truth", "Hydraulic truth · held out"),) + methods
        if args.include_truth_panel
        else methods
    )
    for frame_index in range(frame_count):
        current_scene, sar_positive, sar_history_positive = _sar_scene_masks(
            sar_value,
            sar_acquisition_time,
            float(time_minutes[frame_index]),
        )
        at_acquisition = bool(
            np.isclose(time_minutes[frame_index], sar_acquisition_time[current_scene])
        )
        if at_acquisition:
            acquisition_frame_indices.append(frame_index)
        show_sar = not args.sar_at_acquisition_only or at_acquisition
        adjustment_regions: list[tuple[np.ndarray, np.ndarray, float]] = []
        if args.circle_da_adjustment and at_acquisition:
            adjustment = (
                np.abs(
                    method_mean[traditional_key][frame_index]
                    - method_mean["noda"][frame_index]
                )
                >= 0.50
            )
            adjustment[:river_count] = False
            adjustment_regions = _major_adjustment_regions(
                adjustment,
                xy,
                area,
                edge,
            )
        added_extent = np.zeros(xy.shape[0], dtype=bool)
        added_extent_floodplain = np.zeros(floodplain.sum(), dtype=np.float64)
        added_extent_lattice = None
        if args.outline_da_added_extent and at_acquisition:
            if "da_added_extent_mask" in data:
                added_extent = np.asarray(
                    data["da_added_extent_mask"][frame_index], dtype=bool
                ) & floodplain
            else:
                added_extent = (
                    (method_mean[traditional_key][frame_index] >= wet_threshold)
                    & (method_mean["noda"][frame_index] < wet_threshold)
                    & floodplain
                )
            added_extent_floodplain = added_extent[floodplain].astype(np.float64)
            added_extent_lattice = _native_lattice_mask(
                xy[floodplain],
                added_extent[floodplain],
                floodplain_spacing,
            )
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.25 * len(panels), 6.8),
            dpi=120,
            constrained_layout=True,
        )
        energies = {
            short: _energy(method_member[short][frame_index], truth[frame_index], area)
            for short, _ in methods
        }
        briers = {
            short: _wet_brier(
                method_member[short][frame_index][:, floodplain],
                truth[frame_index, floodplain],
                area[floodplain],
                wet_threshold,
            )
            for short, _ in methods
        }
        truth_wet_floodplain = truth[frame_index, floodplain] > wet_threshold
        false_alarm_area_km2 = {
            short: float(
                np.sum(
                    area[floodplain][~truth_wet_floodplain]
                    * probability[frame_index, floodplain][~truth_wet_floodplain]
                )
                / 1.0e6
            )
            for short, probability in method_probability.items()
        }
        missed_area_km2 = {
            short: float(
                np.sum(
                    area[floodplain][truth_wet_floodplain]
                    * (1.0 - probability[frame_index, floodplain][truth_wet_floodplain])
                )
                / 1.0e6
            )
            for short, probability in method_probability.items()
        }
        best = (
            min(briers, key=briers.get)
            if args.mode in {"extent", "extent-change"}
            else min(energies, key=energies.get)
        )
        truth_wet = truth[frame_index] > wet_threshold
        truth_wet_map = truth_wet & floodplain
        truth_floodplain_area_km2 = float(
            area[floodplain][truth_wet[floodplain]].sum() / 1.0e6
        )

        for axis, (short, label) in zip(axes, panels, strict=True):
            axis.set_facecolor("#f4f2ed")
            axis.scatter(
                xy[floodplain, 0],
                xy[floodplain, 1],
                c=elevation[floodplain],
                cmap="Greys_r",
                norm=terrain_norm,
                s=4.0,
                marker="s",
                linewidths=0,
                alpha=0.20,
                rasterized=True,
            )
            for i, j in zip(source[river_edge], target[river_edge], strict=True):
                axis.plot(
                    xy[[i, j], 0],
                    xy[[i, j], 1],
                    color="#173a5e",
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=2,
                )
            if (
                short != "truth"
                and args.mode != "extent-change"
                and not args.truth_boundary_only
            ):
                axis.scatter(
                    xy[truth_wet_map, 0],
                    xy[truth_wet_map, 1],
                    color="#ef6a61",
                    s=17.0,
                    marker="s",
                    linewidths=0,
                    alpha=0.24,
                    zorder=2.5,
                )
            floodplain_truth = truth_wet[floodplain].astype(np.float64)
            if (
                args.show_initial_truth_boundary
                and initial_truth_floodplain.min() < 0.5 < initial_truth_floodplain.max()
            ):
                axis.tricontour(
                    floodplain_triangulation,
                    initial_truth_floodplain,
                    levels=[0.5],
                    colors=["#4b5563"],
                    linewidths=1.25,
                    linestyles="--",
                    alpha=0.78,
                    zorder=4,
                )
            if (
                not args.hide_truth_boundary
                and floodplain_truth.min() < 0.5 < floodplain_truth.max()
            ):
                axis.tricontour(
                    floodplain_triangulation,
                    floodplain_truth,
                    levels=[0.5],
                    colors=["#111827" if args.mode == "extent-change" else "#df2f2f"],
                    linewidths=2.0,
                    zorder=5,
                )
            if short == "truth":
                depth = truth[frame_index]
                probability = truth_wet.astype(np.float64)
            else:
                depth = method_mean[short][frame_index]
                probability = method_probability[short][frame_index]
            # The 1-D river variables use equivalent storage measures and are
            # not literal 2-D mapped water depths.  Keep them in the separate
            # river-graph overlay, and colour only floodplain control volumes.
            if args.mode == "extent-change":
                current_wet = probability >= 0.5
                if short == "truth":
                    initial_probability = (truth[0] > wet_threshold).astype(np.float64)
                else:
                    initial_probability = method_probability[short][0]
                initial_wet = initial_probability >= 0.5
                persistent = initial_wet & current_wet & floodplain
                newly_wet = ~initial_wet & current_wet & floodplain
                receded = initial_wet & ~current_wet & floodplain
                for mask, colour, size, order in (
                    (persistent, "#9ecae1", 21.0, 3.0),
                    (newly_wet, "#0868ac", 26.0, 4.0),
                    (receded, "#f28e2b", 24.0, 3.5),
                ):
                    axis.scatter(
                        xy[mask, 0],
                        xy[mask, 1],
                        color=colour,
                        s=size,
                        marker="s",
                        linewidths=0,
                        alpha=0.98,
                        zorder=order,
                    )
                # A scalar mappable keeps the shared colorbar API out of this
                # categorical view; the categorical legend carries meaning.
                water = plt.cm.ScalarMappable(norm=norm, cmap="Blues")
            else:
                if args.mode == "extent" and args.binary_extent:
                    wet = (probability >= 0.5) & floodplain
                    water = axis.scatter(
                        xy[wet, 0],
                        xy[wet, 1],
                        color="#2171b5",
                        s=20.0,
                        marker="s",
                        linewidths=0,
                        alpha=0.98,
                        zorder=3,
                    )
                else:
                    wet = (probability >= 0.125) & floodplain
                    if args.mode == "depth-binned":
                        wet = (depth >= wet_threshold) & floodplain
                    water = axis.scatter(
                        xy[wet, 0],
                        xy[wet, 1],
                        c=(probability[wet] if args.mode == "extent" else depth[wet]),
                        cmap=(
                            extent_cmap
                            if args.mode == "extent"
                            else depth_cmap
                            if args.mode == "depth-binned"
                            else "Blues"
                        ),
                        norm=(
                            extent_norm
                            if args.mode == "extent"
                            else depth_binned_norm
                            if args.mode == "depth-binned"
                            else norm
                        ),
                        s=20.0,
                        marker="s",
                        linewidths=0,
                        alpha=0.96,
                        zorder=3,
                    )
            if sar_xy is not None and show_sar:
                sar = np.asarray(sar_xy, dtype=np.float64)
                if not args.hide_sar_history and bool(sar_history_positive.any()):
                    axis.scatter(
                        sar[sar_history_positive, 0],
                        sar[sar_history_positive, 1],
                        marker="x",
                        color="#7a8791",
                        s=17.0,
                        linewidths=0.8,
                        alpha=0.38,
                        zorder=5,
                    )
                axis.scatter(
                    sar[sar_positive, 0],
                    sar[sar_positive, 1],
                    marker="x",
                    color="#00a8b5",
                    s=24.0,
                    linewidths=1.1,
                    alpha=0.80,
                    zorder=6,
                )
            if short == traditional_key and adjustment_regions:
                domain_span = np.array([xmax - xmin, ymax - ymin], dtype=np.float64)
                for lower, upper, _ in adjustment_regions:
                    component_span = upper - lower
                    padded_span = np.maximum(
                        1.18 * component_span,
                        0.055 * domain_span,
                    )
                    axis.add_patch(
                        Ellipse(
                            xy=0.5 * (lower + upper),
                            width=float(padded_span[0]),
                            height=float(padded_span[1]),
                            fill=False,
                            edgecolor="#e31a1c",
                            linewidth=2.8,
                            zorder=8,
                        )
                    )
            if (
                short == traditional_key
                and args.outline_da_added_extent
                and at_acquisition
                and added_extent_lattice is not None
                and added_extent_floodplain.min() < 0.5 < added_extent_floodplain.max()
            ):
                added_x, added_y, added_grid = added_extent_lattice
                axis.contour(
                    added_x,
                    added_y,
                    added_grid,
                    levels=[0.5],
                    colors=["#e31a1c"],
                    linewidths=(3.4 if args.emphasize_small_da_added_cells else 2.0),
                    zorder=8,
                )
                added_count = int(added_extent.sum())
                if args.emphasize_small_da_added_cells and 0 < added_count <= 32:
                    axis.scatter(
                        xy[added_extent, 0],
                        xy[added_extent, 1],
                        marker="s",
                        s=62.0,
                        facecolors="none",
                        edgecolors="#e31a1c",
                        linewidths=2.4,
                        zorder=9,
                    )
                if args.emphasize_small_da_added_cells:
                    axis.text(
                        0.975,
                        0.975,
                        f"SAR update: +{added_count} floodplain cells above {wet_threshold:.2f} m",
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9.5,
                        weight="bold",
                        color="white",
                        bbox={
                            "boxstyle": "round,pad=0.32",
                            "facecolor": "#b91c1c",
                            "edgecolor": "none",
                            "alpha": 0.94,
                        },
                        zorder=10,
                    )
            title_color = "#1261a0" if short == "graph_nao" else "#182633"
            axis.set_title(label, fontsize=15, weight="bold", color=title_color, pad=10)
            if short == "truth":
                axis.text(
                    0.025,
                    0.025,
                    (
                        "Held-out simulator state · unavailable to DA"
                        f"\ntruth area h > {wet_threshold:.3g} m: "
                        f"{truth_floodplain_area_km2:.2f} km²"
                    ),
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=9.5,
                    weight="bold",
                    color="white",
                    bbox={
                        "boxstyle": "round,pad=0.35",
                        "facecolor": "#7f1d1d",
                        "edgecolor": "none",
                        "alpha": 0.92,
                    },
                    zorder=10,
                )
                axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[], yticks=[])
                axis.set_aspect("equal", adjustable="box")
                for spine in axis.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor("#bcc5cc")
                continue
            badge = "BEST  " if short == best else ""
            comparison = ""
            if short == "graph_nao":
                gain = 100.0 * (
                    energies[traditional_key] - energies[short]
                ) / max(
                    energies[traditional_key], np.finfo(float).eps
                )
                wet_gain = 100.0 * (
                    briers[traditional_key] - briers[short]
                ) / max(
                    briers[traditional_key], np.finfo(float).eps
                )
                comparison = (
                    f"\nvs traditional: depth {gain:.1f}% lower · extent {wet_gain:.1f}% lower"
                )
            if args.mode in {"extent", "extent-change"}:
                metric_text = (
                    f"{badge}Wet Brier {briers[short]:.4f} · Depth ES {energies[short]:.4f}"
                    f"\nmiss {missed_area_km2[short]:.2f} km² · false alarm {false_alarm_area_km2[short]:.2f} km²"
                )
            else:
                metric_text = (
                    f"{badge}Depth ES {energies[short]:.4f} · Wet Brier {briers[short]:.4f}"
                )
            axis.text(
                0.025,
                0.025,
                f"{metric_text}{comparison}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5,
                weight="bold" if short == best else "normal",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#1261a0" if short == best else "#263746",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
                zorder=10,
            )
            axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[], yticks=[])
            axis.set_aspect("equal", adjustable="box")
            for spine in axis.spines.values():
                spine.set_linewidth(2.0 if short == "graph_nao" else 0.8)
                spine.set_edgecolor("#1261a0" if short == "graph_nao" else "#bcc5cc")

        phase = (
            f"SAR analysis · acquisition {current_scene + 1}/{len(sar_acquisition_time)}"
            if at_acquisition
            else "physics-only forecast after latest SAR update"
        )
        visual_quantity = (
            "5 cm inundation extent"
            if args.mode == "extent" and args.binary_extent
            else
            (
                "SAR-detectable inundation extent"
                if wet_threshold >= 0.05
                else "Hydraulic wet-support extent"
            )
            if args.mode in {"extent", "extent-change"}
            else (
                "Floodplain water depth"
                if args.include_truth_panel
                else "Ensemble-mean floodplain water depth"
            )
        )
        fig.suptitle(
            f"{visual_quantity} · {event} · {time_minutes[frame_index]:+.0f} min",
            fontsize=20,
            weight="bold",
            color="#132433",
        )
        fig.text(0.5, 0.925, phase, ha="center", fontsize=11.5, color="#52616d")
        fig.text(
            0.5,
            0.885,
            (
                selection_label
                if args.compact_subtitle
                else
                f"{selection_label} · flood-support union crop · same scene per method, grid, ensemble size, and color scale"
                if args.focus_floodplain
                else f"{selection_label} · same scene per method, grid, ensemble size, and color scale"
            ),
            ha="center",
            fontsize=8.5,
            color="#687680",
        )
        handles = (
            [
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#9ecae1", markeredgecolor="none", markersize=9, label="persistently inundated since t=0"),
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#0868ac", markeredgecolor="none", markersize=9, label="newly inundated since t=0"),
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#f28e2b", markeredgecolor="none", markersize=9, label="receded since t=0"),
            ]
            if args.mode == "extent-change"
            else
            [
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#2171b5", markeredgecolor="none", markersize=9, label=f"ensemble-majority inundation: P(h > {wet_threshold:.3g} m) ≥ 0.5")
            ]
            if args.mode == "extent" and args.binary_extent
            else
            [
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#08519c", markeredgecolor="none", markersize=9, label=f"predicted flood: P(h > {wet_threshold:.3g} m) ≥ 0.5"),
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#76b7dd", markeredgecolor="none", markersize=9, label="uncertain flood fringe: 0.125 ≤ P < 0.5"),
            ]
            if args.mode == "extent"
            else [
                Line2D([], [], marker="s", linestyle="none", markerfacecolor="#3182bd", markeredgecolor="none", markersize=9, label="ensemble-mean floodplain water depth")
            ]
        )
        if not args.hide_truth_boundary:
            handles.append(
                Line2D([], [], color="#111827" if args.mode == "extent-change" else "#df2f2f", linewidth=2.0, label="current truth inundation boundary"),
            )
        handles.append(
            Line2D([], [], color="#173a5e", linewidth=2.2, label="river graph")
        )
        if args.show_initial_truth_boundary:
            handles.insert(
                -1,
                Line2D(
                    [],
                    [],
                    color="#4b5563",
                    linewidth=1.25,
                    linestyle="--",
                    label="initial truth boundary (expansion reference)",
                ),
            )
        if sar_xy is not None and show_sar:
            handles.append(
                Line2D([], [], marker="x", linestyle="none", color="#00a8b5", markersize=8, label=f"latest assimilated positive change-SAR detections (scene {current_scene + 1})")
            )
            if current_scene > 0 and not args.hide_sar_history:
                handles.append(
                    Line2D([], [], marker="x", linestyle="none", color="#7a8791", markersize=7, label="earlier positive detections (history)")
                )
        if adjustment_regions:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor="#e31a1c",
                    markeredgewidth=2.2,
                    markersize=10,
                    label="rapid DA adjustment after SAR (|Δh̄| ≥ 0.50 m)",
                )
            )
        if (
            args.outline_da_added_extent
            and at_acquisition
            and added_extent_floodplain.max() > 0.5
        ):
            handles.append(
                Line2D(
                    [],
                    [],
                    color="#e31a1c",
                    linewidth=2.8,
                    label=f"DA-added inundation boundary after SAR (h̄ ≥ {wet_threshold:.2f} m)",
                )
            )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=(
                2
                if args.binary_extent or args.mode == "depth-binned"
                else len(handles)
            ),
            frameon=False,
            bbox_to_anchor=(
                0.5,
                0.002
                if args.binary_extent or args.mode == "depth-binned"
                else -0.015,
            ),
            fontsize=10.5,
        )
        if args.mode != "extent-change" and not (
            args.mode == "extent" and args.binary_extent
        ):
            colorbar = fig.colorbar(water, ax=axes, location="right", shrink=0.78, pad=0.015)
            if args.mode == "extent":
                colorbar.set_ticks((0.3125, 0.75), labels=("uncertain", "predicted"))
            elif args.mode == "depth-binned":
                colorbar.set_ticks(
                    0.5 * (depth_boundaries[:-1] + depth_boundaries[1:]),
                    labels=("0.05–0.20", "0.20–0.50", "0.50–1.00", "1.00–2.00", ">2.00"),
                )
            colorbar.set_label(
                (
                    f"ensemble flood agreement  P(h > {wet_threshold:.3g} m)"
                    if args.mode == "extent"
                    else "floodplain water depth class (m)"
                    if args.mode == "depth-binned"
                    else "floodplain water depth (m)"
                ),
                fontsize=11,
            )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        output_frames.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)

    if not output_frames:
        raise RuntimeError("no frames were rendered")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = round(1000.0 / args.fps)
    durations = [duration] * len(output_frames)
    durations[0] = max(durations[0], 1700)
    durations[-1] = max(durations[-1], 2200)
    if args.circle_da_adjustment or args.outline_da_added_extent:
        for frame_index in acquisition_frame_indices:
            durations[frame_index] = max(durations[frame_index], 1900)
    output_frames[0].save(
        args.output,
        save_all=True,
        append_images=output_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    if args.preview_last is not None:
        args.preview_last.parent.mkdir(parents=True, exist_ok=True)
        output_frames[-1].save(args.preview_last)
    if args.preview_acquisition is not None and acquisition_frame_indices:
        args.preview_acquisition.parent.mkdir(parents=True, exist_ok=True)
        output_frames[acquisition_frame_indices[-1]].save(args.preview_acquisition)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
