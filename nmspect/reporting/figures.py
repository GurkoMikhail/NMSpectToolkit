"""Генератор публикационных научных иллюстраций Matplotlib по сводке потокового исследования `TrackStudySummary`."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.patches as mpatches
import numpy as np
from hepunits import deg, keV, mm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from nmspect.spectrometry.components import fuse_calibrated_windows
from nmspect.tracks.analytics import compute_doi_roc_curve
from nmspect.tracks.study_analyzer import ScatterCalibrationWindowSpec, TrackStudySummary
from nmspect.tracks.taxonomy import (
    DEFAULT_TOPOLOGY_COLORS,
    EXCLUSIVE_TOPOLOGY_NAMES,
    FALLBACK_COLOR_CYCLE,
    TOPOLOGY_LABELS_RU,
)

TOPOLOGY_PALETTE: Final[dict[str, str]] = DEFAULT_TOPOLOGY_COLORS

BAND_PALETTE: Final[tuple[str, ...]] = (
    "#4daf4a",
    "#377eb8",
    "#ff7f00",
    "#984ea3",
    "#e41a1c",
    "#a65628",
    "#f781bf",
)


def _create_agg_figure(fig_width: float, fig_height: float, dpi_val: int = 200) -> Figure:
    """Создаёт изолированный экземпляр `Figure` с рендерером `FigureCanvasAgg` без глобального состояния `pyplot`."""
    fig_obj = Figure(figsize=(fig_width, fig_height), dpi=dpi_val, layout="constrained", facecolor="white")
    FigureCanvasAgg(fig_obj)
    return fig_obj


def _save_figure(fig_obj: Figure, output_path: str | Path, dpi_val: int = 200) -> Path:
    """Сохраняет фигуру в PNG-файл с созданием родительских директорий."""
    resolved_out = Path(output_path)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    fig_obj.savefig(resolved_out, format="png", dpi=dpi_val, facecolor="white")
    return resolved_out


def _resolve_target_window(
    summary: TrackStudySummary,
    window_name: str | None,
) -> ScatterCalibrationWindowSpec:
    """Выбирает запрошенное аналитическое окно из `summary.calibration_windows` (или первое по умолчанию)."""
    if not summary.calibration_windows:
        raise ValueError("В сводке TrackStudySummary не определено ни одного аналитического окна.")
    if window_name is None:
        return summary.calibration_windows[0]
    for win_spec in summary.calibration_windows:
        if win_spec.name == window_name:
            return win_spec
    raise KeyError(f"Окно '{window_name}' отсутствует в summary.calibration_windows.")


def _class_label(summary: TrackStudySummary, class_key: str) -> str:
    """Возвращает человекочитаемую подпись класса таксономии из сводки."""
    if class_key in summary.topology_labels:
        return summary.topology_labels[class_key]
    if class_key in TOPOLOGY_LABELS_RU:
        return TOPOLOGY_LABELS_RU[class_key]
    return class_key


def _class_color(summary: TrackStudySummary, class_key: str, index_hint: int = 0) -> str:
    """Возвращает HEX-цвет класса таксономии из сводки или циклической палитры."""
    if class_key in summary.topology_colors:
        return summary.topology_colors[class_key]
    if class_key in TOPOLOGY_PALETTE:
        return TOPOLOGY_PALETTE[class_key]
    return FALLBACK_COLOR_CYCLE[index_hint % len(FALLBACK_COLOR_CYCLE)]


def plot_experiment_and_topology_schema(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит двухпанельную схему геометрии эксперимента ОФЭКТ и механизмов формирования 10 групп треков."""
    fig_obj = _create_agg_figure(16.8, 8.8, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Геометрия эксперимента ОФЭКТ ({summary.isotope}) и классификация 10 топологических групп треков",
            fontsize=12.5,
            fontweight="bold",
        )

    ax_geom, ax_topo = fig_obj.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.38]})

    # --- Левая панель: трансверсальное сечение гантри и лупа гексагональной решётки ---
    ax_geom.set_aspect("equal")
    ax_geom.set_xlim(-320.0, 320.0)
    ax_geom.set_ylim(-315.0, 315.0)
    ax_geom.axis("off")
    ax_geom.set_title("а) Трансверсальная схема сбора проекций ОФЭКТ", fontsize=10.8, fontweight="bold")

    orbit_circ = mpatches.Circle((0.0, 0.0), radius=250.0, fill=False, ls="--", lw=1.2, edgecolor="#6b7c93")
    ax_geom.add_patch(orbit_circ)

    body_ellipse = mpatches.Ellipse(
        (0.0, 0.0),
        width=190.0,
        height=145.0,
        facecolor="#dcebfa",
        edgecolor="#1f5aa6",
        lw=2.0,
    )
    ax_geom.add_patch(body_ellipse)
    for ang_idx, sphere_radius in enumerate((6.0, 8.0, 10.0, 13.0, 16.0, 21.0)):
        ang_val = float(ang_idx) * np.pi / 3.0
        pos_u = 45.0 * np.cos(ang_val)
        pos_v = 36.0 * np.sin(ang_val)
        hot_sphere = mpatches.Circle(
            (pos_u, pos_v),
            radius=sphere_radius * 0.55,
            facecolor="#ff6b4a",
            edgecolor="#991f00",
            lw=1.0,
        )
        ax_geom.add_patch(hot_sphere)

    head_positions = (1.0,) if summary.num_detectors == 1 else (1.0, -1.0)
    for idx_head, sign_y in enumerate(head_positions, start=1):
        base_y = sign_y * 235.0
        coll_rect = mpatches.Rectangle(
            (-165.0, base_y - (18.0 if sign_y > 0 else 0.0)),
            330.0,
            18.0,
            facecolor="#8c9ba5",
            edgecolor="#2c3e50",
            lw=1.4,
        )
        cryst_rect = mpatches.Rectangle(
            (-165.0, base_y + (0.0 if sign_y > 0 else -12.0)),
            330.0,
            12.0,
            facecolor="#b8e994",
            edgecolor="#1e824c",
            lw=1.4,
        )
        back_rect = mpatches.Rectangle(
            (-170.0, base_y + (12.0 if sign_y > 0 else -34.0)),
            340.0,
            22.0,
            facecolor="#e8dff5",
            edgecolor="#6c5ce7",
            lw=1.2,
        )
        ax_geom.add_patch(coll_rect)
        ax_geom.add_patch(cryst_rect)
        ax_geom.add_patch(back_rect)
        label_y = base_y + (44.0 if sign_y > 0 else -48.0)
        ax_geom.text(
            0.0,
            label_y,
            f"Детектор {idx_head}: коллиматор + кристалл NaI(Tl) + световод/ФЭУ",
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#1f2d3d",
        )

    ax_geom.annotate(
        "",
        xy=(0.0, 216.0),
        xytext=(0.0, 25.0),
        arrowprops={"arrowstyle": "->", "color": "#1b9e77", "lw": 2.2},
    )
    ax_geom.text(
        -195.0,
        -108.0,
        f"Радионуклид: {summary.isotope}\nРакурсов: {summary.num_views}\nДетекторов: {summary.num_detectors}",
        ha="left",
        va="center",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )

    # Лупа гексагональной решётки со строгим соотношением сторон 1:1
    ax_inset = ax_geom.inset_axes((0.68, 0.04, 0.29, 0.29))
    ax_inset.set_aspect("equal")
    ax_inset.set_xlim(-2.4, 2.4)
    ax_inset.set_ylim(-2.4, 2.4)
    ax_inset.axis("off")
    lens_bg = mpatches.Circle((0.0, 0.0), radius=2.25, facecolor="#94a3b8", edgecolor="#1e293b", lw=1.8)
    ax_inset.add_patch(lens_bg)
    hex_offsets = (
        (0.0, 0.0),
        (1.05, 0.0),
        (-1.05, 0.0),
        (0.525, 0.91),
        (-0.525, 0.91),
        (0.525, -0.91),
        (-0.525, -0.91),
    )
    for center_hx, center_hy in hex_offsets:
        hex_patch = mpatches.RegularPolygon(
            (center_hx, center_hy),
            numVertices=6,
            radius=0.46,
            orientation=0.0,
            facecolor="white",
            edgecolor="#1e293b",
            lw=1.1,
        )
        ax_inset.add_patch(hex_patch)
    ax_geom.text(215.0, -295.0, "Профиль септ (1:1)", ha="center", va="center", fontsize=8.0, color="#334155")

    # --- Правая панель: слои траекторий и все 10 дизъюнктных классов C1..C10 ---
    num_classes = len(summary.exclusive_class_names)
    ax_topo.set_xlim(0.0, 100.0)
    ax_topo.set_ylim(0.0, 100.0)
    ax_topo.axis("off")
    ax_topo.set_title(
        f"б) Формирование {num_classes} взаимоисключающих топологических групп (100% баланс)",
        fontsize=10.8,
        fontweight="bold",
    )

    zone_specs = (
        (78.0, 16.0, "#ede7f6", "#673ab7", "Задетекторная зона (световод ФЭУ, пирекс, защитный кожух)"),
        (58.0, 16.0, "#e8f5e9", "#2e7d32", "Сцинтилляционный кристалл NaI(Tl) (фотоэффект / вылет кванта)"),
        (36.0, 18.0, "#eceff1", "#455a64", "Свинцовый многоканальный коллиматор (каналы и септы)"),
        (8.0, 22.0, "#e3f2fd", "#1565c0", "Объём фантома / источника (прямой вылет, Комптон, Рэлей)"),
    )
    for zone_y, zone_h, fill_col, edge_col, zone_title in zone_specs:
        zone_box = mpatches.FancyBboxPatch(
            (3.0, zone_y),
            52.0,
            zone_h,
            boxstyle="round,pad=0.6",
            facecolor=fill_col,
            edgecolor=edge_col,
            lw=1.4,
        )
        ax_topo.add_patch(zone_box)
        ax_topo.text(5.0, zone_y + zone_h - 3.2, zone_title, fontsize=8.0, fontweight="bold", color="#1e293b")

    # Схематические физические траектории переноса гамма-квантов по горизонтальной шкале слоёв
    track_paths = (
        ("C1", "#1b9e77", [(8.0, 16.0), (8.0, 66.0)], "-"),
        ("C2", "#66c2a5", [(13.0, 16.0), (13.0, 65.0), (15.5, 76.0)], "--"),
        ("C3", "#d95f02", [(18.5, 16.0), (22.0, 66.0)], "-"),
        ("C4", "#fc8d62", [(23.5, 16.0), (27.0, 65.0), (29.0, 76.0)], "--"),
        ("C5", "#377eb8", [(30.0, 14.0), (33.0, 23.0), (33.0, 66.0)], "-"),
        ("C6", "#6baed6", [(35.5, 13.0), (38.0, 19.0), (36.5, 25.0), (36.5, 66.0)], "-"),
        ("C7", "#0288d1", [(40.5, 14.0), (41.5, 23.0), (41.5, 66.0)], ":"),
        ("C8", "#984ea3", [(45.0, 16.0), (45.0, 44.0), (47.0, 66.0)], "-"),
        ("C9", "#bc80bd", [(48.5, 14.0), (50.0, 23.0), (50.0, 45.0), (51.5, 66.0)], "-"),
        ("C10", "#e41a1c", [(53.0, 16.0), (53.0, 86.0), (50.5, 66.0)], "-"),
    )
    for code_lbl, col_hex, pts_seq, style_str in track_paths:
        xs_arr = [pt[0] for pt in pts_seq]
        ys_arr = [pt[1] for pt in pts_seq]
        ax_topo.plot(xs_arr, ys_arr, color=col_hex, lw=1.9, ls=style_str)
        ax_topo.scatter([xs_arr[-1]], [ys_arr[-1]], color=col_hex, s=22, zorder=5)
        ax_topo.text(xs_arr[0], 9.5, code_lbl, ha="center", va="center", fontsize=7.2, fontweight="bold", color=col_hex)

    fractions_full = summary.global_topology_fractions("full_spectrum")
    legend_y = 92.5
    legend_step = min(8.6, 84.0 / float(max(num_classes, 1)))
    ax_topo.text(58.0, 96.5, "Доли в полном спектре:", fontsize=8.8, fontweight="bold", color="#0f172a")
    for idx_cls, topo_key in enumerate(summary.exclusive_class_names):
        pct_val = 100.0 * fractions_full[topo_key]
        short_lbl = _class_label(summary, topo_key)
        col_hex = _class_color(summary, topo_key, idx_cls)
        swatch = mpatches.Rectangle((58.0, legend_y - 1.4), 2.3, 2.6, facecolor=col_hex, edgecolor="none")
        ax_topo.add_patch(swatch)
        ax_topo.text(
            61.2,
            legend_y,
            f"{short_lbl} — {pct_val:.2f}%",
            va="center",
            fontsize=7.6,
            color="#1e293b",
        )
        legend_y -= legend_step

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_topology_spectra(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит энергетический спектр детектора с декомпозицией на 10 топологических классов (линейная и лог. шкалы)."""
    fig_obj = _create_agg_figure(16.4, 6.8, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Декомпозиция энергетического спектра ({summary.isotope}) по 10 топологическим классам треков",
            fontsize=12.0,
            fontweight="bold",
        )

    ax_lin, ax_log = fig_obj.subplots(1, 2)
    edges_display = summary.arrays["spectrum_edges"] / float(keV)
    centers_display = 0.5 * (edges_display[:-1] + edges_display[1:])
    spec_total = summary.arrays["spec_total"].astype(np.float64)

    for axis_item, is_log in ((ax_lin, False), (ax_log, True)):
        axis_item.plot(centers_display, np.maximum(spec_total, 0.5 if is_log else 0.0), color="#111111", lw=1.8, label="Полный спектр")
        for idx_cls, topo_key in enumerate(summary.exclusive_class_names):
            comp_spec = summary.arrays[f"spec_{topo_key}"].astype(np.float64)
            if float(np.sum(comp_spec)) <= 0.0:
                continue
            plot_vals = np.maximum(comp_spec, 0.5) if is_log else comp_spec
            axis_item.plot(
                centers_display,
                plot_vals,
                color=_class_color(summary, topo_key, idx_cls),
                lw=1.3,
                label=_class_label(summary, topo_key),
            )
        for win_spec in summary.calibration_windows:
            low_disp = win_spec.main_lower / float(keV)
            high_disp = win_spec.main_upper / float(keV)
            axis_item.axvspan(low_disp, high_disp, color="#ffd54f", alpha=0.22)

        axis_item.set_xlabel("Измеренная энергия вспышки E, кэВ", fontsize=9.5)
        axis_item.set_ylabel("Число зарегистрированных треков на бин", fontsize=9.5)
        axis_item.grid(True, linestyle=":", alpha=0.5)
        if is_log:
            axis_item.set_yscale("log")
            axis_item.set_ylim(bottom=0.8)
            axis_item.set_title("б) Логарифмическая шкала (высокоэнергетические хвосты и вылет)", fontsize=10.2)
            axis_item.legend(loc="upper right", fontsize=7.2, framealpha=0.92)
        else:
            low_max = max(float(edges_display[-1]) * 0.35, 120.0)
            if summary.calibration_windows:
                max_win_edge = max(win_s.main_upper / float(keV) for win_s in summary.calibration_windows)
                low_max = min(float(edges_display[-1]), max(110.0, max_win_edge * 1.45))
            axis_item.set_xlim(0.0, low_max)
            axis_item.set_title("а) Область рабочих фотопиков и окон DEW/TEW", fontsize=10.2)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_emission_band_spectra(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит разложение спектра по полосам начальной эмиссии и 2D-матрицу перекрёстного вклада (100% баланс)."""
    fig_obj = _create_agg_figure(16.6, 6.8, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Вклад энергетических полос рождения {summary.isotope} и матрица перекрёстного загрязнения фотопика",
            fontsize=12.0,
            fontweight="bold",
        )

    ax_spec, ax_mat = fig_obj.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.15]})
    edges_display = summary.arrays["spectrum_edges"] / float(keV)
    centers_display = 0.5 * (edges_display[:-1] + edges_display[1:])
    spec_total = np.maximum(summary.arrays["spec_total"].astype(np.float64), 0.5)

    ax_spec.plot(centers_display, spec_total, color="#111111", lw=1.8, label="Суммарный спектр")
    for idx_band, band_cfg in enumerate(summary.emission_bands):
        band_spec = summary.arrays[f"spec_{band_cfg.name}"].astype(np.float64)
        if float(np.sum(band_spec)) <= 0.0:
            continue
        col_hex = BAND_PALETTE[idx_band % len(BAND_PALETTE)]
        lbl_str = band_cfg.label if band_cfg.label else band_cfg.name
        ax_spec.plot(centers_display, np.maximum(band_spec, 0.5), color=col_hex, lw=1.45, label=lbl_str)

    ax_spec.set_yscale("log")
    ax_spec.set_ylim(bottom=0.8)
    ax_spec.set_xlabel("Измеренная энергия вспышки E, кэВ", fontsize=9.5)
    ax_spec.set_ylabel("Число треков на бин (лог. шкала)", fontsize=9.5)
    ax_spec.set_title("а) Спектры от отдельных полос начальной энергии рождения", fontsize=10.2)
    ax_spec.grid(True, linestyle=":", alpha=0.5)
    ax_spec.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    target_win = _resolve_target_window(summary, None)
    win_total = max(summary.window_totals[target_win.name], 1)
    band_topo_dict = summary.window_band_topology_counts[target_win.name]

    if summary.exclusive_class_names == EXCLUSIVE_TOPOLOGY_NAMES:
        grouped_columns: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("Прямой\n(C1+C2)", ("geometric_full", "geometric_escape")),
            ("Септы\n(C3+C4)", ("septal_full", "septal_escape")),
            ("Комптон 1x\n(C5)", ("phantom_compton_single",)),
            ("Многокр.\n(C6)", ("phantom_inelastic_multi",)),
            ("Рэлей\n(C7)", ("phantom_coherent_only",)),
            ("Коллим.\n(C8+C9)", ("collimator_only_scatter", "phantom_and_collimator")),
            ("Backscatter\n(C10)", ("backscatter",)),
        )
    else:
        grouped_columns = tuple(
            (_class_label(summary, cls_key)[:18], (cls_key,))
            for cls_key in summary.exclusive_class_names
        )

    mat_pct = np.zeros((len(summary.emission_bands), len(grouped_columns)), dtype=np.float64)
    for row_idx, band_cfg in enumerate(summary.emission_bands):
        for col_idx, (_, keys_grp) in enumerate(grouped_columns):
            sub_cnt = sum(int(band_topo_dict[band_cfg.name][topo_k]) for topo_k in keys_grp)
            mat_pct[row_idx, col_idx] = 100.0 * float(sub_cnt) / float(win_total)

    disp_mat = np.maximum(mat_pct, 1e-3)
    im_obj = ax_mat.imshow(disp_mat, cmap="YlOrRd", norm=LogNorm(vmin=1e-2, vmax=max(10.0, float(np.max(disp_mat)))), aspect="auto")
    for row_idx in range(mat_pct.shape[0]):
        for col_idx in range(mat_pct.shape[1]):
            cell_val = mat_pct[row_idx, col_idx]
            txt_str = "—" if cell_val < 0.01 else (f"{cell_val:.2f}%" if cell_val < 10.0 else f"{cell_val:.1f}%")
            txt_col = "white" if cell_val >= 5.0 else "#1e293b"
            ax_mat.text(col_idx, row_idx, txt_str, ha="center", va="center", fontsize=8.0, color=txt_col)

    ax_mat.set_xticks(np.arange(len(grouped_columns)))
    ax_mat.set_xticklabels([col_lbl for col_lbl, _ in grouped_columns], fontsize=8.2)
    ax_mat.set_yticks(np.arange(len(summary.emission_bands)))
    ax_mat.set_yticklabels(
        [bnd.label if bnd.label else bnd.name for bnd in summary.emission_bands],
        fontsize=8.2,
    )
    ax_mat.set_title(f"б) Матрица вкладов в {target_win.label} (сумма = 100%)", fontsize=10.2)
    cbar = fig_obj.colorbar(im_obj, ax=ax_mat, fraction=0.042, pad=0.03)
    cbar.set_label("Доля счёта окна, %", fontsize=8.5)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_angular_psf_components(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    window_name: str | None = None,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит 1D профили угловой функции рассеяния точки (ФРТ) по компонентам и результат вычитания TEW."""
    target_win = _resolve_target_window(summary, window_name)
    fig_obj = _create_agg_figure(16.5, 6.6, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Угловая функция рассеяния точки (ФРТ) в окне {target_win.label} и коррекция фона TEW",
            fontsize=12.0,
            fontweight="bold",
        )

    ax_core, ax_wings = fig_obj.subplots(1, 2)
    fine_edges = summary.arrays["psf_fine_edges"]
    fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    wide_edges = summary.arrays["psf_wide_edges"]
    wide_centers = 0.5 * (wide_edges[:-1] + wide_edges[1:])

    w_key = target_win.name
    fine_total = summary.arrays[f"psf_fine_{w_key}_total"].astype(np.float64)
    fine_geom = summary.arrays[f"psf_fine_{w_key}_geometric_full"].astype(np.float64)

    ax_core.plot(fine_centers, fine_total, color="#111111", lw=2.0, label="Суммарная ФРТ окна (MEW)")
    ax_core.plot(fine_centers, fine_geom, color="#1b9e77", lw=2.0, ls="--", label="Эталон C1: прямой геометрический")

    if target_win.has_tew and f"psf_fine_{w_key}_low_sub_total" in summary.arrays:
        wt_low, wt_high = target_win.tew_standard_weights()
        low_prof = summary.arrays[f"psf_fine_{w_key}_low_sub_total"].astype(np.float64)
        high_prof = summary.arrays[f"psf_fine_{w_key}_high_sub_total"].astype(np.float64)
        k_opt = summary.optimal_calibration_factors[w_key]["optimal_k_tew"]
        tew_std_prof = np.maximum(0.0, fine_total - (wt_low * low_prof + wt_high * high_prof))
        tew_opt_prof = np.maximum(0.0, fine_total - k_opt * (wt_low * low_prof + wt_high * high_prof))
        ax_core.plot(fine_centers, tew_std_prof, color="#ff7f00", lw=1.6, ls="-.", label="После стандартного TEW (k = 1)")
        ax_core.plot(
            fine_centers,
            tew_opt_prof,
            color="#6a3d9a",
            lw=1.8,
            label=f"После калиброванного TEW (k = {k_opt:.2f})",
        )

    ax_core.set_xlabel("Поперечный угловой наклон tan(θ_u)", fontsize=9.5)
    ax_core.set_ylabel("Число треков на бин", fontsize=9.5)
    ax_core.set_title("а) Центральное ядро угловой ФРТ и точность вычитания TEW", fontsize=10.2)
    ax_core.grid(True, linestyle=":", alpha=0.5)
    ax_core.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    wide_total = np.maximum(summary.arrays[f"psf_wide_{w_key}_total"].astype(np.float64), 0.5)
    ax_wings.plot(wide_centers, wide_total, color="#111111", lw=1.9, label="Полная ФРТ")
    wing_keys = (
        (
            "geometric_full",
            "septal_full",
            "septal_escape",
            "phantom_compton_single",
            "collimator_only_scatter",
            "backscatter",
        )
        if summary.exclusive_class_names == EXCLUSIVE_TOPOLOGY_NAMES
        else summary.exclusive_class_names[:8]
    )
    for idx_cls, topo_key in enumerate(wing_keys):
        arr_key = f"psf_wide_{w_key}_{topo_key}"
        if arr_key in summary.arrays:
            sub_prof = summary.arrays[arr_key].astype(np.float64)
            if float(np.sum(sub_prof)) > 0.0:
                ax_wings.plot(
                    wide_centers,
                    np.maximum(sub_prof, 0.5),
                    color=_class_color(summary, topo_key, idx_cls),
                    lw=1.4,
                    label=_class_label(summary, topo_key),
                )

    ax_wings.set_yscale("log")
    ax_wings.set_ylim(bottom=0.8)
    ax_wings.set_xlabel("Поперечный угловой наклон tan(θ_u)", fontsize=9.5)
    ax_wings.set_ylabel("Число треков на бин (лог. шкала)", fontsize=9.5)
    ax_wings.set_title("б) Далёкие негауссовы крылья угловой ФРТ по классам", fontsize=10.2)
    ax_wings.grid(True, linestyle=":", alpha=0.5)
    ax_wings.legend(loc="upper right", fontsize=7.4, framealpha=0.92)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_psf_2d_and_anisotropy(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    window_name: str | None = None,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит 2D карты угловой ФРТ `(tan_u, tan_v)` и азимутальный профиль гексагонального пробоя септ."""
    target_win = _resolve_target_window(summary, window_name)
    w_key = target_win.name
    fig_obj = _create_agg_figure(16.8, 5.8, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Двумерная угловая ФРТ и азимутальная анизотропия пробоя септ ({target_win.label})",
            fontsize=12.0,
            fontweight="bold",
        )

    ax_tot, ax_septa, ax_az = fig_obj.subplots(1, 3)
    edges_2d = summary.arrays["psf_2d_edges"]
    extent_2d = [float(edges_2d[0]), float(edges_2d[-1]), float(edges_2d[0]), float(edges_2d[-1])]

    map_tot = np.maximum(summary.arrays[f"psf2d_{w_key}_total"].astype(np.float64), 0.5)
    map_septa = np.maximum(
        (
            summary.arrays[f"psf2d_{w_key}_septal_full"] + summary.arrays[f"psf2d_{w_key}_septal_escape"]
        ).astype(np.float64),
        0.5,
    )

    for axis_map, mat_data, map_title in (
        (ax_tot, map_tot, "а) Суммарная 2D ФРТ (лог. шкала)"),
        (ax_septa, map_septa, "б) Компонента пробоя септ (C3 + C4)"),
    ):
        im_map = axis_map.imshow(
            mat_data.T,
            origin="lower",
            extent=extent_2d,
            cmap="inferno",
            norm=LogNorm(vmin=0.5, vmax=max(5.0, float(np.max(mat_data)))),
            aspect="equal",
        )
        axis_map.set_xlabel("Поперечный наклон tan(θ_u)", fontsize=9.2)
        axis_map.set_ylabel("Аксиальный наклон tan(θ_v)", fontsize=9.2)
        axis_map.set_title(map_title, fontsize=10.0)
        fig_obj.colorbar(im_map, ax=axis_map, fraction=0.046, pad=0.03)

    az_edges_disp = summary.arrays["azimuth_edges"] / float(deg)
    az_centers_disp = 0.5 * (az_edges_disp[:-1] + az_edges_disp[1:])

    for az_tag, lbl_str, col_hex in (
        ("septal_escape", "Пробой септ с вылетом (C4)", "#fc8d62"),
        ("septal_full", "Прямой пробой септ (C3)", "#d95f02"),
        ("phantom_compton_all", "Комптон в фантоме (изотропный)", "#377eb8"),
    ):
        az_hist = summary.arrays[f"azimuth_{w_key}_{az_tag}"].astype(np.float64)
        mean_az = float(np.mean(az_hist))
        if mean_az > 0.0:
            ax_az.plot(az_centers_disp, az_hist / mean_az, color=col_hex, lw=1.7, label=lbl_str)

    ax_az.axhline(1.0, color="#666666", ls="--", lw=1.0)
    ax_az.set_xlim(0.0, 360.0)
    ax_az.set_xticks([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    ax_az.set_xlabel("Азимутальный угол ψ в плоскости детектора, град", fontsize=9.2)
    ax_az.set_ylabel("Нормированный профиль N(ψ) / <N>", fontsize=9.2)
    ax_az.set_title("в) Азимутальная модуляция лучей септ (шаг 60°)", fontsize=10.0)
    ax_az.grid(True, linestyle=":", alpha=0.5)
    az_handles, _ = ax_az.get_legend_handles_labels()
    if az_handles:
        ax_az.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_angular_calibration_profiles(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    window_name: str | None = None,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит поракурсные профили доли рассеяния, коэффициентов калибровки `k(phi)` и доли обнуляемых пикселей."""
    target_win = _resolve_target_window(summary, window_name)
    w_key = target_win.name
    fig_obj = _create_agg_figure(16.6, 6.2, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Угловая динамика рассеяния и калибровочных множителей по {summary.num_views} ракурсам ({target_win.label})",
            fontsize=12.0,
            fontweight="bold",
        )

    ax_sf, ax_k = fig_obj.subplots(1, 2)
    angles_disp = summary.arrays["view_angles"] / float(deg)
    sort_order = np.argsort(angles_disp)
    sorted_angles = angles_disp[sort_order]

    sf_true = 100.0 * summary.arrays[f"view_{w_key}_true_scatter_fraction"][sort_order]
    sf_tew = 100.0 * summary.arrays[f"view_{w_key}_tew_std_fraction"][sort_order]
    sf_dew = 100.0 * summary.arrays[f"view_{w_key}_dew_std_fraction"][sort_order]
    neg_tew = 100.0 * summary.arrays[f"view_{w_key}_neg_pixel_fraction_tew"][sort_order]

    ax_sf.plot(sorted_angles, sf_true, color="#1b9e77", lw=2.0, label="Истинная доля негеометрического фона, %")
    if target_win.has_tew:
        ax_sf.plot(sorted_angles, sf_tew, color="#d95f02", lw=1.7, ls="--", label="Оценка стандартного TEW (k = 1), %")
        ax_sf.plot(sorted_angles, neg_tew, color="#e41a1c", lw=1.4, ls=":", label="Доля пикселей с перевычитанием TEW < 0, %")
    if target_win.has_dew:
        ax_sf.plot(sorted_angles, sf_dew, color="#377eb8", lw=1.6, ls="-.", label="Оценка стандартного DEW (k = 0.5), %")

    ax_sf.set_xlabel("Угловое положение детекторной головки φ, град", fontsize=9.5)
    ax_sf.set_ylabel("Доля событий / пикселей, %", fontsize=9.5)
    ax_sf.set_title("а) Сравнение истинного фона и оконных оценок по ракурсам", fontsize=10.2)
    ax_sf.grid(True, linestyle=":", alpha=0.5)
    ax_sf.legend(loc="best", fontsize=8.0, framealpha=0.92)

    k_tew_arr = summary.arrays[f"view_{w_key}_optimal_k_tew"][sort_order]
    k_dew_arr = summary.arrays[f"view_{w_key}_optimal_k_dew"][sort_order]
    if target_win.has_tew:
        k_tew_global = summary.optimal_calibration_factors[w_key]["optimal_k_tew"]
        ax_k.plot(
            sorted_angles,
            k_tew_arr,
            color="#6a3d9a",
            lw=2.0,
            label=f"Оптимальный множитель трапеции k_TEW(φ) (глоб. = {k_tew_global:.3f})",
        )
    if target_win.has_dew:
        k_dew_global = summary.optimal_calibration_factors[w_key]["optimal_k_dew"]
        ax_k.plot(
            sorted_angles,
            k_dew_arr,
            color="#1f78b4",
            lw=1.8,
            ls="--",
            label=f"Оптимальный множитель k_DEW(φ) (глоб. = {k_dew_global:.3f})",
        )
    ax_k.axhline(1.0, color="#666666", ls=":", lw=1.0, label="Номинальный уровень трапеции k_TEW = 1.0")
    ax_k.set_xlabel("Угловое положение детекторной головки φ, град", fontsize=9.5)
    ax_k.set_ylabel("Безразмерный калибровочный коэффициент k", fontsize=9.5)
    ax_k.set_title("б) Угловая устойчивость оптимальных коэффициентов вычитания фона", fontsize=10.2)
    ax_k.grid(True, linestyle=":", alpha=0.5)
    ax_k.legend(loc="best", fontsize=8.0, framealpha=0.92)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_doi_and_roc(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    window_name: str | None = None,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит профили глубины взаимодействия (DOI) в кристалле и ROC-кривые аппаратной селекции `z_hat <= z_max`."""
    target_win = _resolve_target_window(summary, window_name)
    w_key = target_win.name
    fig_obj = _create_agg_figure(16.4, 6.2, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Распределение глубины взаимодействия (DOI) и ROC-селективность в окне {target_win.label}",
            fontsize=12.0,
            fontweight="bold",
        )

    ax_doi, ax_roc = fig_obj.subplots(1, 2)
    depth_edges = summary.arrays["depth_edges"]
    depth_centers_disp = 0.5 * (depth_edges[:-1] + depth_edges[1:]) / float(mm)

    sig_hist = summary.arrays[f"depth_{w_key}_geometric_full"].astype(np.float64)
    back_hist = summary.arrays[f"depth_{w_key}_backscatter"].astype(np.float64)
    sepe_hist = summary.arrays[f"depth_{w_key}_septal_escape"].astype(np.float64)
    comp_hist = summary.arrays[f"depth_{w_key}_phantom_compton_all"].astype(np.float64)

    for hist_arr, lbl_str, col_hex in (
        (sig_hist, "C1: прямой геометрический сигнал", "#1b9e77"),
        (comp_hist, "C5+C6: комптоновское рассеяние в фантоме", "#377eb8"),
        (sepe_hist, "C4: пробой септ с вылетом из кристалла", "#fc8d62"),
        (back_hist, "C10: задетекторное обратное рассеяние", "#e41a1c"),
    ):
        hist_sum = float(np.sum(hist_arr))
        if hist_sum > 0.0:
            ax_doi.plot(depth_centers_disp, hist_arr / hist_sum, color=col_hex, lw=1.8, label=lbl_str)

    ax_doi.set_xlabel("Глубина энерговыделения от входного окна кристалла z, мм", fontsize=9.5)
    ax_doi.set_ylabel("Нормированная плотность вероятности на бин", fontsize=9.5)
    ax_doi.set_title("а) Профили глубины вспышек по физическим механизмам", fontsize=10.2)
    ax_doi.grid(True, linestyle=":", alpha=0.5)
    if ax_doi.get_legend_handles_labels()[0]:
        ax_doi.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    tot_depth = summary.arrays[f"depth_{w_key}_total"].astype(np.float64)
    hard_bg_hist = back_hist + sepe_hist
    if float(np.sum(back_hist)) > 0.0:
        eval_bg = back_hist
    elif float(np.sum(hard_bg_hist)) > 0.0:
        eval_bg = hard_bg_hist
    else:
        eval_bg = np.maximum(0.0, tot_depth - sig_hist)

    if float(np.sum(sig_hist)) > 0.0 and float(np.sum(eval_bg)) > 0.0:
        for res_val, col_hex, style_str, prefix_lbl in (
            (0.0 * mm, "#1b9e77", "-", "Идеальный DOI (FWHM = 0 мм)"),
            (1.5 * mm, "#377eb8", "--", "Высокое разрешение (FWHM = 1,5 мм)"),
            (3.0 * mm, "#d95f02", "-.", "Умеренное разрешение (FWHM = 3,0 мм)"),
        ):
            roc_res = compute_doi_roc_curve(
                signal_depth_hist=sig_hist,
                background_depth_hist=eval_bg,
                depth_edges=depth_edges,
                depth_resolution_fwhm=res_val,
            )
            ax_roc.plot(
                100.0 * (1.0 - roc_res.roc_fpr),
                100.0 * roc_res.roc_tpr,
                color=col_hex,
                ls=style_str,
                lw=2.0,
                label=f"{prefix_lbl}: AUC = {roc_res.auc:.4f}",
            )

    ax_roc.set_xlim(0.0, 102.0)
    ax_roc.set_ylim(0.0, 102.0)
    ax_roc.set_xlabel("Подавление задетекторного фона (1 − FPR), %", fontsize=9.5)
    ax_roc.set_ylabel("Сохранение геометрического сигнала TPR, %", fontsize=9.5)
    ax_roc.set_title("б) Семейство ROC-кривых аппаратного DOI-дискриминатора", fontsize=10.2)
    ax_roc.grid(True, linestyle=":", alpha=0.5)
    if ax_roc.get_legend_handles_labels()[0]:
        ax_roc.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


def plot_planar_projections_grid(
    summary: TrackStudySummary,
    output_path: str | Path,
    *,
    window_name: str | None = None,
    show_suptitle: bool = False,
    dpi: int = 200,
) -> Path:
    """Строит сравнение планарных 2D-проекций опорного ракурса (MEW, Primary, TEW и Dual-Peak слияние)."""
    target_win = _resolve_target_window(summary, window_name)
    w_key = target_win.name
    fig_obj = _create_agg_figure(16.8, 5.6, dpi_val=dpi)
    if show_suptitle:
        fig_obj.suptitle(
            f"Планарные 2D-проекции опорного ракурса и калиброванная коррекция рассеяния ({summary.isotope})",
            fontsize=12.0,
            fontweight="bold",
        )

    edges_u = summary.arrays["proj_u_edges"] / float(mm)
    edges_v = summary.arrays["proj_v_edges"] / float(mm)
    extent_uv = [float(edges_u[0]), float(edges_u[-1]), float(edges_v[0]), float(edges_v[-1])]

    proj_tot = summary.arrays[f"ref_proj_{w_key}_total"].astype(np.float64)
    proj_pri = summary.arrays[f"ref_proj_{w_key}_primary"].astype(np.float64)
    proj_low = summary.arrays[f"ref_proj_{w_key}_low_sub"].astype(np.float64)
    proj_high = summary.arrays[f"ref_proj_{w_key}_high_sub"].astype(np.float64)

    wt_low, wt_high = target_win.tew_standard_weights()
    scat_est = wt_low * proj_low + wt_high * proj_high if target_win.has_tew else target_win.dew_standard_weight() * proj_low
    k_opt = (
        summary.optimal_calibration_factors[w_key]["optimal_k_tew"]
        if target_win.has_tew
        else summary.optimal_calibration_factors[w_key]["optimal_k_dew"]
    )
    proj_corr = fuse_calibrated_windows([proj_tot], [scat_est], calibration_factors=[k_opt])

    if len(summary.calibration_windows) >= 2:
        win_list_tot = []
        win_list_scat = []
        win_list_k = []
        for win_item in summary.calibration_windows:
            sub_tot = summary.arrays[f"ref_proj_{win_item.name}_total"].astype(np.float64)
            sub_low = summary.arrays[f"ref_proj_{win_item.name}_low_sub"].astype(np.float64)
            sub_high = summary.arrays[f"ref_proj_{win_item.name}_high_sub"].astype(np.float64)
            wl_i, wh_i = win_item.tew_standard_weights()
            sub_scat = wl_i * sub_low + wh_i * sub_high if win_item.has_tew else win_item.dew_standard_weight() * sub_low
            sub_k = (
                summary.optimal_calibration_factors[win_item.name]["optimal_k_tew"]
                if win_item.has_tew
                else summary.optimal_calibration_factors[win_item.name]["optimal_k_dew"]
            )
            win_list_tot.append(sub_tot)
            win_list_scat.append(sub_scat)
            win_list_k.append(sub_k)
        fourth_proj = fuse_calibrated_windows(win_list_tot, win_list_scat, calibration_factors=win_list_k)
        fourth_title = "г) Слияние калиброванных окон Dual-Peak"
    else:
        fourth_proj = summary.arrays[f"ref_proj_{w_key}_non_geometric"].astype(np.float64)
        fourth_title = "г) Истинный негеометрический фон окна"

    panels = (
        (proj_tot, f"а) Без коррекции MEW ({int(np.sum(proj_tot)):,} имп.)"),
        (proj_pri, f"б) Эталон Primary C1 ({int(np.sum(proj_pri)):,} имп.)"),
        (proj_corr, f"в) Калиброванная коррекция (k = {k_opt:.2f})"),
        (fourth_proj, fourth_title),
    )

    axes_row = fig_obj.subplots(1, 4)
    for axis_item, (mat_img, panel_title) in zip(axes_row, panels, strict=True):
        pos_vals = mat_img[mat_img > 0.0]
        vmax_val = float(np.percentile(pos_vals, 99.5)) if pos_vals.size > 0 else 1.0
        im_panel = axis_item.imshow(
            mat_img.T,
            origin="lower",
            extent=extent_uv,
            cmap="inferno",
            vmin=0.0,
            vmax=max(vmax_val, 1.0),
            aspect="equal",
        )
        axis_item.set_xlabel("Поперечная ось u, мм", fontsize=8.8)
        axis_item.set_ylabel("Аксиальная ось v, мм", fontsize=8.8)
        axis_item.set_title(panel_title, fontsize=9.2)
        fig_obj.colorbar(im_panel, ax=axis_item, fraction=0.046, pad=0.03)

    return _save_figure(fig_obj, output_path, dpi_val=dpi)


__all__ = [
    "BAND_PALETTE",
    "TOPOLOGY_PALETTE",
    "plot_angular_calibration_profiles",
    "plot_angular_psf_components",
    "plot_doi_and_roc",
    "plot_emission_band_spectra",
    "plot_experiment_and_topology_schema",
    "plot_planar_projections_grid",
    "plot_psf_2d_and_anisotropy",
    "plot_topology_spectra",
]
