"""Автоматический генератор научно-технических отчётов ОФЭКТ (`.docx`, `.pdf`, `.doc`) по сводке `TrackStudySummary`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hepunits import keV, mm

from nmspect.reporting.converters import DocumentConversionResult, DocumentFormatConverter
from nmspect.reporting.docx_builder import PageGeometryConfig, ScientificDocxBuilder
from nmspect.reporting.figures import (
    plot_angular_calibration_profiles,
    plot_angular_psf_components,
    plot_doi_and_roc,
    plot_emission_band_spectra,
    plot_experiment_and_topology_schema,
    plot_planar_projections_grid,
    plot_psf_2d_and_anisotropy,
    plot_topology_spectra,
)
from nmspect.tracks.analytics import compute_doi_roc_curve
from nmspect.tracks.study_analyzer import TrackStudySummary
from nmspect.tracks.taxonomy import EXCLUSIVE_TOPOLOGY_NAMES, TOPOLOGY_LABELS_RU


@dataclass(frozen=True)
class ReportMetadataConfig:
    """Паспортные реквизиты научно-технического отчёта ОФЭКТ."""

    document_label: str = "НАУЧНО-ТЕХНИЧЕСКИЙ ОТЧЁТ"
    subtitle: str = (
        "Векторная реконструкция траекторий гамма-квантов, декомпозиция спектров и функции рассеяния точки (ФРТ), "
        "калибровка вычитания рассеянного излучения и оценка селективности по глубине взаимодействия"
    )
    organization: str = "Программный комплекс NMSpectToolkit"
    author: str = ""
    figure_dpi: int = 200


@dataclass(frozen=True)
class GeneratedStudyReport:
    """Сводный результат сборки научно-технического отчёта и сопутствующих иллюстраций."""

    docx_path: Path
    pdf_path: Path | None
    legacy_doc_path: Path | None
    figure_paths: dict[str, Path]
    conversion: DocumentConversionResult | None


def _fmt_int(value: float) -> str:
    """Форматирует целое число с неразрывными пробелами между разрядами."""
    return f"{round(value):,}".replace(",", " ")


def _fmt_num(value: float, digits: int = 2) -> str:
    """Форматирует вещественное число в русской академической записи с запятой."""
    return f"{value:.{digits}f}".replace(".", ",")


class TrackStudyReportGenerator:
    """Строит комплект публикационных иллюстраций и собирает академический отчёт `.docx`/`.pdf`/`.doc`."""

    def __init__(
        self,
        report_config: ReportMetadataConfig | None = None,
        page_geometry: PageGeometryConfig | None = None,
        converter: DocumentFormatConverter | None = None,
    ) -> None:
        """Инициализирует генератор научного отчёта."""
        self.report_config: ReportMetadataConfig = (
            report_config if report_config is not None else ReportMetadataConfig()
        )
        self.page_geometry: PageGeometryConfig = (
            page_geometry if page_geometry is not None else PageGeometryConfig()
        )
        self.converter: DocumentFormatConverter = (
            converter if converter is not None else DocumentFormatConverter()
        )

    def generate_figures(
        self,
        summary: TrackStudySummary,
        figures_dir: str | Path,
        *,
        show_suptitle: bool = False,
    ) -> dict[str, Path]:
        """Генерирует полный набор PNG-графиков исследования в указанной директории."""
        out_dir = Path(figures_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dpi_val = self.report_config.figure_dpi
        primary_win = summary.calibration_windows[0].name if summary.calibration_windows else None

        produced: dict[str, Path] = {
            "schema": plot_experiment_and_topology_schema(
                summary,
                out_dir / "fig01_experiment_and_topology_schema.png",
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "topology_spectra": plot_topology_spectra(
                summary,
                out_dir / "fig02_topology_spectra.png",
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "emission_bands": plot_emission_band_spectra(
                summary,
                out_dir / "fig03_emission_bands_and_matrix.png",
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "angular_psf": plot_angular_psf_components(
                summary,
                out_dir / "fig04_angular_psf_components.png",
                window_name=primary_win,
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "psf_2d_anisotropy": plot_psf_2d_and_anisotropy(
                summary,
                out_dir / "fig05_psf_2d_and_anisotropy.png",
                window_name=primary_win,
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "angular_calibration": plot_angular_calibration_profiles(
                summary,
                out_dir / "fig06_angular_calibration_profiles.png",
                window_name=primary_win,
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "doi_and_roc": plot_doi_and_roc(
                summary,
                out_dir / "fig07_doi_and_roc.png",
                window_name=primary_win,
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
            "planar_projections": plot_planar_projections_grid(
                summary,
                out_dir / "fig08_planar_projections.png",
                window_name=primary_win,
                show_suptitle=show_suptitle,
                dpi=dpi_val,
            ),
        }
        return produced

    def generate_report(
        self,
        summary: TrackStudySummary,
        output_docx_path: str | Path,
        *,
        figures_dir: str | Path | None = None,
        export_pdf: bool = False,
        export_legacy_doc: bool = False,
        strict_conversion: bool = False,
    ) -> GeneratedStudyReport:
        """Формирует полный научно-технический отчёт `.docx` и опционально экспортирует `.pdf` / `.doc`."""
        target_docx = Path(output_docx_path).resolve()
        resolved_fig_dir = (
            Path(figures_dir).resolve()
            if figures_dir is not None
            else target_docx.parent / f"{target_docx.stem}_figures"
        )
        fig_paths = self.generate_figures(summary, resolved_fig_dir, show_suptitle=False)

        doc = ScientificDocxBuilder(geometry=self.page_geometry)
        cfg = self.report_config

        # --- Титульная полоса и аннотация ---
        doc.add_paragraph(
            cfg.organization,
            align="center",
            first_line_indent=0.0 * mm,
            space_before=60,
            space_after=180,
            bold=True,
            font_half_points=24,
        )
        doc.add_paragraph(
            cfg.document_label,
            align="center",
            first_line_indent=0.0 * mm,
            space_before=120,
            space_after=80,
            bold=True,
            font_half_points=32,
        )
        doc.add_paragraph(
            summary.study_title,
            align="center",
            first_line_indent=0.0 * mm,
            space_before=40,
            space_after=80,
            bold=True,
            font_half_points=28,
        )
        doc.add_paragraph(
            cfg.subtitle,
            align="center",
            first_line_indent=0.0 * mm,
            space_before=20,
            space_after=160,
            italic=True,
            font_half_points=23,
        )
        if cfg.author:
            doc.add_paragraph(
                f"Исполнитель: {cfg.author}",
                align="center",
                first_line_indent=0.0 * mm,
                space_before=20,
                space_after=120,
                font_half_points=23,
            )

        num_classes = len(summary.exclusive_class_names)
        doc.add_paragraph(
            f"**Аннотация.** В настоящем отчёте представлены результаты сквозного траекторного и спектрометрического "
            f"анализа Монте-Карло моделирования однофотонной эмиссионной компьютерной томографии (ОФЭКТ) "
            f"для радионуклида **{summary.isotope}**. Обработано **{_fmt_int(summary.num_files)}** входных файлов "
            f"симуляции (**{_fmt_int(summary.num_views)}** проекционных ракурсов при числе детекторных головок "
            f"**{summary.num_detectors}**), объединяющих **{_fmt_int(summary.total_tracks)}** зарегистрированных "
            f"треков частиц и **{_fmt_int(summary.total_points)}** пространственных узлов взаимодействия. "
            f"Выполнена строгая взаимоисключающая классификация треков на {num_classes} физических групп (100%-ный баланс счёта), "
            f"проведена декомпозиция энергетических спектров и угловой функции рассеяния точки (ФРТ), определены "
            f"оптимальные калибровочные коэффициенты оконной коррекции рассеяния DEW/TEW и оценена эффективность "
            f"селекции по глубине взаимодействия в сцинтилляционном кристалле (DOI)."
        )

        doc.add_toc_field()
        doc.add_page_break()

        # --- Раздел 1: Геометрия эксперимента и математические алгоритмы ---
        doc.add_heading("1. Постановка задачи и алгоритмы векторного анализа треков", level=1)
        doc.add_paragraph(
            "Реконструкция индивидуальных историй переноса фотонов выполняется в компактном представлении "
            "разреженных строк CSR (`TrackBatch`), где каждая полилиния трека $m$ объединяет начальное состояние "
            "эмиссии и упорядоченную по времени последовательность узлов взаимодействия в фантоме, коллиматоре, "
            "кристалле и задетекторном окружении. Для каждой детекторной головки из матрицы позы $4 \\times 4$ "
            "извлекается правый ортонормированный репер $(\\mathbf{e}_u, \\mathbf{e}_v, \\mathbf{e}_n)$ с внешней "
            "нормалью $\\mathbf{e}_n$, поворачиваемый вокруг оси гантри на угол ракурса $\\Delta\\varphi$:"
        )
        doc.add_latex_equation(
            r"\tan\theta_u = \frac{(\mathbf{r}_{\text{det}} - \mathbf{r}_0)\cdot \mathbf{e}_u}"
            r"{|(\mathbf{r}_{\text{det}} - \mathbf{r}_0)\cdot \mathbf{e}_n|}, \quad "
            r"\tan\theta_v = \frac{(\mathbf{r}_{\text{det}} - \mathbf{r}_0)\cdot \mathbf{e}_v}"
            r"{|(\mathbf{r}_{\text{det}} - \mathbf{r}_0)\cdot \mathbf{e}_n|}",
            equation_number=1,
        )
        doc.add_paragraph(
            "где $\\mathbf{r}_0$ — мировая точка рождения кванта, а $\\mathbf{r}_{\\text{det}}$ — энерго взвешенный "
            "центроид энерговыделения внутри кристалла. Трассировка прохождения нерассеянного фотона через "
            "многоканальный коллиматор выполняется путём точного аналитического пересечения луча с входной и "
            "выходной плоскостями свинцовых каналов и кубического округления аксиальных гексагональных индексов "
            "$(q_1, q_2)$. Это позволяет отделить прямое геометрическое прохождение внутри одного отверстия от "
            "косого пробоя свинцовых септ."
        )
        doc.add_paragraph(
            f"Полное пространство зарегистрированных событий $\\Omega$ разбивается на {num_classes} попарно непересекающихся "
            f"топологических классов $C_1, \\dots, C_{{{num_classes}}}$, удовлетворяющих точному тождеству сохранения числа треков:"
        )
        doc.add_latex_equation(
            rf"\Omega = \bigsqcup_{{m=1}}^{{{num_classes}}} C_m, \quad \sum_{{m=1}}^{{{num_classes}}} |C_m| = N_{{\text{{total}}}}",
            equation_number=2,
        )
        doc.add_figure(
            fig_paths["schema"],
            f"Рисунок 1. Схема геометрии сбора данных ОФЭКТ ({summary.isotope}, {summary.num_views} ракурсов) "
            f"и формирование {num_classes} взаимоисключающих топологических групп треков.",
        )

        # --- Раздел 2: Глобальный баланс классов и энергетические спектры ---
        doc.add_heading(f"2. Энергетические спектры и баланс {num_classes} топологических классов", level=1)
        doc.add_paragraph(
            f"Энергетический отклик сцинтилляционного тракта моделируется гауссовым размытием суммарного "
            f"энерговыделения в кристалле с дисперсией $\\sigma_E(E) \\propto \\sqrt{{E}}$. Ниже приведена сводная "
            f"таблица строго взаимоисключающего распределения треков по {num_classes} топологическим классам в полном спектре "
            f"и основных фотопиковых окнах."
        )

        primary_win = summary.calibration_windows[0] if summary.calibration_windows else None
        secondary_win = summary.calibration_windows[1] if len(summary.calibration_windows) >= 2 else None

        headers_bal = ["Топологический класс траектории", "Полный спектр, треков (%)"]
        weights_bal = [38.0, 22.0]
        if primary_win is not None:
            headers_bal.append(f"{primary_win.label}, треков (%)")
            weights_bal.append(20.0)
        if secondary_win is not None:
            headers_bal.append(f"{secondary_win.label}, треков (%)")
            weights_bal.append(20.0)

        tot_full = max(summary.window_totals["full_spectrum"], 1)
        rows_bal: list[list[str]] = []
        for topo_key in summary.exclusive_class_names:
            cnt_f = summary.window_topology_counts["full_spectrum"][topo_key]
            pct_f = 100.0 * float(cnt_f) / float(tot_full)
            cls_label = summary.topology_labels.get(topo_key, TOPOLOGY_LABELS_RU.get(topo_key, topo_key))
            row_cells = [cls_label, f"{_fmt_int(cnt_f)} ({_fmt_num(pct_f)}%)"]
            for opt_win in (primary_win, secondary_win):
                if opt_win is not None:
                    tot_w = max(summary.window_totals[opt_win.name], 1)
                    cnt_w = summary.window_topology_counts[opt_win.name][topo_key]
                    pct_w = 100.0 * float(cnt_w) / float(tot_w)
                    row_cells.append(f"{_fmt_int(cnt_w)} ({_fmt_num(pct_w)}%)")
            rows_bal.append(row_cells)

        summary_row_bal = [
            f"Итого по всем {num_classes} классам (баланс)",
            f"{_fmt_int(summary.window_totals['full_spectrum'])} (100,00%)",
        ]
        for opt_win in (primary_win, secondary_win):
            if opt_win is not None:
                summary_row_bal.append(f"{_fmt_int(summary.window_totals[opt_win.name])} (100,00%)")
        rows_bal.append(summary_row_bal)

        doc.add_table(
            f"Таблица 1. Распределение зарегистрированных треков по {num_classes} дизъюнктным топологическим группам",
            headers=headers_bal,
            rows=rows_bal,
            column_weights=weights_bal,
            summary_rows=(-1,),
        )

        doc.add_figure(
            fig_paths["topology_spectra"],
            f"Рисунок 2. Энергетический спектр {summary.isotope}, разложенный на {num_classes} топологических классов треков "
            "(а — область рабочих окон в линейном масштабе; б — полный диапазон в логарифмическом масштабе).",
        )
        doc.add_figure(
            fig_paths["emission_bands"],
            "Рисунок 3. Спектральные вклады отдельных полос начальной энергии эмиссии и двумерная матрица "
            "перекрёстного формирования счёта в основном фотопиковом окне.",
        )

        # --- Раздел 3: Функция рассеяния точки (ФРТ) и гексагональная анизотропия ---
        doc.add_heading("3. Декомпозиция угловой функции рассеяния точки (ФРТ) и анизотропия септ", level=1)
        doc.add_paragraph(
            "Пространственная разрешающая способность коллимированного детектора определяется угловой функцией "
            "рассеяния точки (ФРТ) в координатах $(\\tan\\theta_u, \\tan\\theta_v)$. В то время как прямая "
            "геометрическая компонента $C_1$ локализована внутри апертуры единичного канала коллиматора, рассеянные "
            "в фантоме и коллиматоре фотоны, а также кванты пробоя септ и обратного рассеяния формируют протяжённые "
            "негауссовы крылья ФРТ."
        )

        if primary_win is not None:
            w_name = primary_win.name
            psf_rows: list[list[str]] = []
            if summary.exclusive_class_names == EXCLUSIVE_TOPOLOGY_NAMES:
                psf_table_items: tuple[tuple[str, str], ...] = (
                    ("geometric_full", TOPOLOGY_LABELS_RU["geometric_full"]),
                    ("septal_full", TOPOLOGY_LABELS_RU["septal_full"]),
                    ("septal_escape", TOPOLOGY_LABELS_RU["septal_escape"]),
                    ("phantom_compton_single", TOPOLOGY_LABELS_RU["phantom_compton_single"]),
                    ("phantom_inelastic_multi", TOPOLOGY_LABELS_RU["phantom_inelastic_multi"]),
                    ("collimator_only_scatter", TOPOLOGY_LABELS_RU["collimator_only_scatter"]),
                    ("backscatter", TOPOLOGY_LABELS_RU["backscatter"]),
                    ("total", "Суммарный поток в окне"),
                )
            else:
                psf_table_items = (
                    *(
                        (cls_k, summary.topology_labels.get(cls_k, cls_k))
                        for cls_k in summary.exclusive_class_names
                    ),
                    ("total", "Суммарный поток в окне"),
                )

            for grp_key, grp_label in psf_table_items:
                rms_u = math.sqrt(max(0.0, summary.window_mean_tan_sq[w_name][grp_key]))
                rms_r = math.sqrt(max(0.0, summary.window_mean_tan_radial_sq[w_name][grp_key]))
                mean_z = summary.window_mean_depth[w_name][grp_key] / float(mm)
                mean_sep = summary.window_mean_septa[w_name][grp_key]
                psf_rows.append(
                    [
                        grp_label,
                        _fmt_num(rms_u, 4),
                        _fmt_num(rms_r, 4),
                        _fmt_num(mean_z, 2),
                        _fmt_num(mean_sep, 2),
                    ]
                )

            doc.add_table(
                f"Таблица 2. Параметры угловой ФРТ, средней глубины взаимодействия и пересечения септ ({primary_win.label})",
                headers=(
                    "Физическая компонента",
                    "СКО наклона tan(θ_u)",
                    "Радиальное СКО tan(θ_r)",
                    "Средняя глубина DOI, мм",
                    "Среднее число септ",
                ),
                rows=psf_rows,
                column_weights=(36.0, 16.0, 16.0, 16.0, 16.0),
                summary_rows=(-1,),
            )

        doc.add_figure(
            fig_paths["angular_psf"],
            "Рисунок 4. Поперечные профили угловой функции рассеяния точки (ФРТ): центральное ядро до и после "
            "вычитания фона TEW (а) и широкие логарифмические крылья по топологическим классам (б).",
        )
        doc.add_figure(
            fig_paths["psf_2d_anisotropy"],
            "Рисунок 5. Двумерные карты угловой ФРТ и шестилучевая азимутальная модуляция пробоя гексагональных септ.",
        )

        # --- Раздел 4: Калибровка оконного вычитания рассеяния и планарные проекции ---
        doc.add_heading("4. Калибровка коррекции рассеяния DEW/TEW и качество планарных проекций", level=1)
        doc.add_paragraph(
            "Стандартная трапецеидальная оценка Огавы (TEW) аппроксимирует подложку негеометрического фона в основном "
            "фотопиковом окне шириной $W_M$ линейной интерполяцией счётов в нижнем ($N_L$, ширина $W_L$) и верхнем "
            "($N_H$, ширина $W_H$) боковых подокнах. При наличии близко расположенных мультиплетов или крутого спада "
            "комптоновского континуума стандартная трапеция ($k_{\\text{TEW}} = 1$) систематически смещает оценку "
            "фона, поэтому вводится калиброванный протокол вычитания и многопикового слияния окон:"
        )
        doc.add_latex_equation(
            r"\hat{P}_w = \left[ M_w - k_w^{\text{opt}} \left( \frac{W_{M,w}}{2 W_{L,w}} L_w + "
            r"\frac{W_{M,w}}{2 W_{H,w}} H_w \right) \right]_+, \quad "
            r"P_{\text{fused}} = \sum_{w} \alpha_w \hat{P}_w",
            equation_number=3,
        )

        cal_rows: list[list[str]] = []
        for win_spec in summary.calibration_windows:
            w_tot = max(summary.window_totals[win_spec.name], 1)
            w_geom = summary.primary_counts(win_spec.name)
            true_sf_pct = 100.0 * float(w_tot - w_geom) / float(w_tot)
            opt_map = summary.optimal_calibration_factors[win_spec.name]
            bounds_str = f"{_fmt_num(win_spec.main_lower / keV, 1)}–{_fmt_num(win_spec.main_upper / keV, 1)}"
            cal_rows.append(
                [
                    win_spec.label,
                    bounds_str,
                    _fmt_num(100.0 - true_sf_pct, 2),
                    _fmt_num(true_sf_pct, 2),
                    _fmt_num(opt_map["optimal_k_dew"], 3),
                    _fmt_num(opt_map["optimal_k_tew"], 3),
                ]
            )
        if cal_rows:
            doc.add_table(
                "Таблица 3. Чистота энергетических окон и оптимальные калибровочные коэффициенты вычитания фона",
                headers=(
                    "Энергетическое окно",
                    "Границы, кэВ",
                    "Доля C1 Primary, %",
                    "Негеометрический фон, %",
                    "Опт. множитель k_DEW",
                    "Опт. множитель k_TEW",
                ),
                rows=cal_rows,
                column_weights=(26.0, 14.0, 15.0, 15.0, 15.0, 15.0),
            )

        doc.add_figure(
            fig_paths["angular_calibration"],
            f"Рисунок 6. Поракурсные зависимости доли рассеянного фона и оптимальных коэффициентов калибровки "
            f"по {summary.num_views} угловым положениям детекторов.",
        )
        doc.add_figure(
            fig_paths["planar_projections"],
            "Рисунок 7. Сравнение планарных 2D-проекций опорного ракурса до коррекции (MEW), эталона прямого "
            "излучения (Primary) и калиброванной оконной фильтрации.",
        )

        # --- Раздел 5: Глубина взаимодействия в кристалле (DOI) и выводы ---
        doc.add_heading("5. Селекция по глубине взаимодействия в кристалле (DOI) и выводы", level=1)
        doc.add_paragraph(
            "Низкоэнергетические первичные гамма-кванты поглощаются преимущественно в тонком входном слое "
            "кристалла NaI(Tl), тогда как жёсткие кванты обратного рассеяния от световода ФЭУ ($C_{10}$) и "
            "прострела септ с вылетом ($C_4$) распределены по всей толщине сцинтиллятора со смещением к тыльной "
            "грани. Аппаратное ограничение оценки глубины $\\hat{z} \\le z_{\\max}$ обеспечивает эффективное "
            "подавление жёсткого дальнодействующего фона."
        )

        if primary_win is not None:
            w_key = primary_win.name
            sig_h = summary.arrays[f"depth_{w_key}_geometric_full"].astype(np.float64)
            bkg_h = summary.arrays[f"depth_{w_key}_backscatter"].astype(np.float64)
            if float(np.sum(bkg_h)) <= 0.0:
                bkg_h = (
                    summary.arrays[f"depth_{w_key}_backscatter"] + summary.arrays[f"depth_{w_key}_septal_escape"]
                ).astype(np.float64)
            if float(np.sum(bkg_h)) <= 0.0:
                bkg_h = np.maximum(0.0, summary.arrays[f"depth_{w_key}_total"].astype(np.float64) - sig_h)

            if float(np.sum(sig_h)) > 0.0 and float(np.sum(bkg_h)) > 0.0:
                dep_edges = summary.arrays["depth_edges"]
                roc_rows: list[list[str]] = []
                for res_fwhm, mode_lbl in (
                    (0.0 * mm, "Идеальная локализация глубины (0,0 мм)"),
                    (1.5 * mm, "Кремниевые ФЭУ высокого разрешения (1,5 мм)"),
                    (3.0 * mm, "Стандартная оценка ФЭУ Ангера (3,0 мм)"),
                ):
                    roc_res = compute_doi_roc_curve(sig_h, bkg_h, dep_edges, depth_resolution_fwhm=res_fwhm)
                    opt_idx = int(np.argmin(np.abs(roc_res.thresholds - roc_res.optimal_threshold)))
                    roc_rows.append(
                        [
                            mode_lbl,
                            _fmt_num(roc_res.auc, 4),
                            _fmt_num(roc_res.optimal_threshold / mm, 2),
                            _fmt_num(100.0 * float(roc_res.signal_efficiency[opt_idx]), 2),
                            _fmt_num(100.0 * float(roc_res.background_rejection[opt_idx]), 2),
                        ]
                    )
                doc.add_table(
                    f"Таблица 4. Показатели ROC-анализа DOI-дискриминатора в окне {primary_win.label}",
                    headers=(
                        "Режим разрешения по глубине FWHM",
                        "Площадь под кривой AUC",
                        "Оптимальный порог z_max, мм",
                        "Сохранение сигнала Primary, %",
                        "Подавление фона, %",
                    ),
                    rows=roc_rows,
                    column_weights=(32.0, 17.0, 17.0, 17.0, 17.0),
                )

        doc.add_figure(
            fig_paths["doi_and_roc"],
            "Рисунок 8. Распределения глубины взаимодействия в кристалле по топологическим классам и семейство "
            "ROC-кривых DOI-селекции при различном аппаратном разрешении.",
        )

        crosstalk_pct = (
            100.0 * float(summary.shared_particles) / float(max(summary.unique_particles, 1))
        )
        doc.add_bullet(
            f"Строгая {num_classes}-классовая таксономия обеспечила полный баланс счёта по всем {_fmt_int(summary.total_tracks)} "
            f"трекам; доля междетекторного перелёта между головками составила {_fmt_num(crosstalk_pct, 4)}% от "
            f"{_fmt_int(summary.unique_particles)} уникальных зарегистрированных частиц."
        )
        if primary_win is not None:
            opt_k = summary.optimal_calibration_factors[primary_win.name]["optimal_k_tew"]
            doc.add_bullet(
                f"В основном аналитическом окне {primary_win.label} калибровка трапеции Огавы со множителем "
                f"$k_{{\\text{{TEW}}}}^{{\\text{{opt}}}} = {_fmt_num(opt_k, 3)}$ устраняет систематическое смещение "
                f"стандартного метода TEW и подавляет ложное обнуление периферийных пикселей проекции."
            )

        saved_docx = doc.save_docx(target_docx)

        conversion_res: DocumentConversionResult | None = None
        pdf_target = saved_docx.with_suffix(".pdf") if export_pdf else None
        doc_target = saved_docx.with_suffix(".doc") if export_legacy_doc else None
        if pdf_target is not None or doc_target is not None:
            conversion_res = self.converter.convert_all(
                saved_docx,
                pdf_path=pdf_target,
                legacy_doc_path=doc_target,
                strict=strict_conversion,
            )

        return GeneratedStudyReport(
            docx_path=saved_docx,
            pdf_path=conversion_res.pdf_path if conversion_res is not None else None,
            legacy_doc_path=conversion_res.legacy_doc_path if conversion_res is not None else None,
            figure_paths=fig_paths,
            conversion=conversion_res,
        )


__all__ = [
    "GeneratedStudyReport",
    "ReportMetadataConfig",
    "TrackStudyReportGenerator",
]
