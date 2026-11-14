"""CLI-точка входа (`nmspect-tracks`) для потокового анализа 3D-треков, экспорта VTK и сборки научного отчёта."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hepunits import mm

from nmspect.exporters.vtk_tracks import VtkTrackExporter
from nmspect.reporting import ReportMetadataConfig, TrackStudyReportGenerator
from nmspect.spectrometry.presets import normalize_isotope_name
from nmspect.tracks.study_analyzer import (
    StreamingTrackStudyAnalyzer,
    TrackStudyConfig,
    TrackStudySummary,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для конвейера анализа треков и генерации отчёта."""
    parser = argparse.ArgumentParser(
        prog="nmspect-tracks",
        description=(
            "Потоковый статистический анализ 3D-треков гамма-квантов ОФЭКТ (10 классов таксономии, "
            "спектры, ФРТ, калибровка DEW/TEW, DOI), экспорт в ParaView VTK и генерация отчёта .docx/.pdf/.doc."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Путь к директории с входными файлами HDF5 или к одиночному файлу симуляции.",
    )
    parser.add_argument(
        "--bundle-json",
        type=str,
        default=None,
        help="Путь к ранее сохранённому JSON-файлу сводки (для пересборки графиков/отчёта без повторного чтения HDF5).",
    )
    parser.add_argument(
        "--bundle-npz",
        type=str,
        default=None,
        help="Путь к ранее сохранённому NPZ-архиву массивов сводки (в паре с --bundle-json).",
    )
    parser.add_argument(
        "--isotope",
        type=str,
        default="161Tb",
        help="Радионуклид исследования ('161Tb' или '99mTc', по умолчанию: 161Tb).",
    )
    parser.add_argument(
        "--study-title",
        type=str,
        default=None,
        help="Пользовательский заголовок исследования для сводки и титульного листа отчёта.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Опциональное ограничение числа обрабатываемых файлов HDF5.",
    )
    parser.add_argument(
        "--energy-resolution",
        type=float,
        default=0.099,
        help="Относительное энергетическое разрешение детектора FWHM на опорной энергии (по умолчанию: 0.099).",
    )
    parser.add_argument(
        "--spatial-resolution",
        type=float,
        default=3.8,
        help="Внутреннее пространственное разрешение кристалла FWHM в мм (по умолчанию: 3.8).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=4.0,
        help="Размер пикселя опорных 2D-проекций в мм (по умолчанию: 4.0).",
    )
    parser.add_argument(
        "--vtk-sample-per-view",
        type=int,
        default=200,
        help="Число треков на один ракурс в стратифицированной выборке для ParaView VTK (по умолчанию: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно генератора случайных чисел (по умолчанию: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/track_study",
        help="Выходная директория для сводки, VTK-сцены и научного отчёта (по умолчанию: results/track_study).",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default=None,
        help="Префикс имён создаваемых выходных файлов.",
    )
    parser.add_argument(
        "--export-vtk",
        action="store_true",
        help="Экспортировать мультиблочную 3D-сцену треков .vtm для ParaView.",
    )
    parser.add_argument(
        "--build-report",
        action="store_true",
        help="Сгенерировать публикационные графики и научно-технический отчёт в формате .docx.",
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Дополнительно экспортировать отчёт в формат .pdf (подразумевает --build-report).",
    )
    parser.add_argument(
        "--export-doc",
        action="store_true",
        help="Дополнительно экспортировать отчёт в формат Word 97-2003 .doc (подразумевает --build-report).",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="",
        help="ФИО автора / исполнителя для титульного блока отчёта.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI-команды `nmspect-tracks`."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_isotope = normalize_isotope_name(args.isotope)
    resolved_prefix = (
        args.study_prefix
        if args.study_prefix is not None
        else f"{canonical_isotope.lower()}_track_study"
    )

    if args.bundle_json is not None or args.bundle_npz is not None:
        if args.bundle_json is None or args.bundle_npz is None:
            parser.error("Аргументы --bundle-json и --bundle-npz должны передаваться одновременно.")
        summary = TrackStudySummary.load_bundle(args.bundle_json, args.bundle_npz)
    else:
        if args.input_path is None:
            parser.error("Необходимо указать --input-path либо пару --bundle-json / --bundle-npz.")
        config = TrackStudyConfig.for_isotope(
            isotope=canonical_isotope,
            study_title=args.study_title,
            energy_resolution=float(args.energy_resolution),
            intrinsic_resolution=float(args.spatial_resolution) * mm,
            projection_pixel_size=float(args.pixel_size) * mm,
            vtk_sample_per_view=max(0, int(args.vtk_sample_per_view)),
            random_seed=int(args.seed),
        )
        analyzer = StreamingTrackStudyAnalyzer(config=config)
        src_path = Path(args.input_path)
        if src_path.is_dir():
            summary = analyzer.analyze_directory(src_path, max_files=args.max_files)
        elif src_path.is_file():
            summary = analyzer.analyze_files([src_path])
        else:
            raise FileNotFoundError(f"Путь к входным данным не найден: {src_path}")

        json_out, npz_out = summary.save_bundle(
            out_dir / f"{resolved_prefix}_summary.json",
            out_dir / f"{resolved_prefix}_arrays.npz",
        )
        print(f"Сводка исследования сохранена: {json_out} | {npz_out}")

    if args.export_vtk and summary.sampled_tracks is not None and summary.sampled_tracks.num_tracks > 0:
        vtk_exporter = VtkTrackExporter()
        vtm_path = vtk_exporter.export_scene_vtm(
            tracks=summary.sampled_tracks,
            metadata=summary.metadata,
            output_path=out_dir / f"{resolved_prefix}_tracks.vtm",
        )
        print(f"3D-сцена треков ParaView сохранена: {vtm_path}")

    need_report = bool(args.build_report or args.export_pdf or args.export_doc)
    if need_report:
        report_gen = TrackStudyReportGenerator(
            report_config=ReportMetadataConfig(author=str(args.author)),
        )
        report_res = report_gen.generate_report(
            summary=summary,
            output_docx_path=out_dir / f"{resolved_prefix}_report.docx",
            figures_dir=out_dir / f"{resolved_prefix}_figures",
            export_pdf=bool(args.export_pdf),
            export_legacy_doc=bool(args.export_doc),
        )
        print(f"Научно-технический отчёт (.docx): {report_res.docx_path}")
        if report_res.pdf_path is not None:
            print(f"Версия в формате PDF (.pdf):     {report_res.pdf_path}")
        if report_res.legacy_doc_path is not None:
            print(f"Версия Word 97-2003 (.doc):      {report_res.legacy_doc_path}")
        if report_res.conversion is not None and not report_res.conversion.succeeded:
            print(f"Примечание конвертера: {report_res.conversion.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
