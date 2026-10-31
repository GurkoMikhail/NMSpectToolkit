"""CLI-точка входа (`nmspect-process`) для запуска конвейера постобработки сырых HDF5-данных ОФЭКТ."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hepunits import mm, ns

from nmspect.pipelines.projection_pipeline import run_isotope_nema_pipeline
from nmspect.spectrometry.presets import format_presets_summary, normalize_isotope_name


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для конвейера обработки проекций."""
    parser = argparse.ArgumentParser(
        prog="nmspect-process",
        description="Постобработка сырых HDF5-данных симуляций ОФЭКТ (161Tb / 99mTc), спектрометрия и экспорт.",
    )
    parser.add_argument(
        "--isotope",
        type=str,
        default="161Tb",
        help="Радионуклид исследования ('161Tb', 'tb161', '99mTc', 'tc99m'). По умолчанию: 161Tb.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Путь к директории с входными HDF5-файлами ракурсов (по умолчанию из data/raw/).",
    )
    parser.add_argument(
        "--preset",
        dest="presets",
        action="append",
        default=None,
        help="Ключ пресета энергетических окон (можно указать несколько раз; по умолчанию все пресеты изотопа).",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Вывести список всех доступных пресетов энергетических окон и завершить работу.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=4.2,
        help="Размер пикселя проекционной матрицы в мм (по умолчанию: 4.2).",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=None,
        help="Явный радиус орбиты вращения детекторов в мм (по умолчанию извлекается из метаданных HDF5).",
    )
    parser.add_argument(
        "--time-resolution",
        type=float,
        default=230.0,
        help="Временное разрешение сцинтиллятора в нс (по умолчанию: 230.0).",
    )
    parser.add_argument(
        "--energy-resolution",
        type=float,
        default=0.095,
        help="Относительное энергетическое разрешение FWHM детектора (по умолчанию: 0.095).",
    )
    parser.add_argument(
        "--spatial-resolution",
        type=float,
        default=3.8,
        help="Внутреннее пространственное разрешение FWHM кристалла в мм (по умолчанию: 3.8).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results",
        help="Корневая директория для сохранения результатов (по умолчанию: results).",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default=None,
        help="Префикс имён создаваемых файлов исследования.",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Не экспортировать вспомогательные форматы (.npy, .spr, .png), оставить только NeXus/DICOM/ParaView.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Зерно генератора случайных чисел для воспроизводимого размытия.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI-команды `nmspect-process`."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print(format_presets_summary())
        return 0

    canonical_isotope = normalize_isotope_name(args.isotope)
    orbit_radius_val = args.orbit_radius * mm if args.orbit_radius is not None else None

    exported = run_isotope_nema_pipeline(
        isotope=canonical_isotope,
        data_dir=args.data_dir,
        preset_keys=args.presets,
        pixel_size=float(args.pixel_size) * mm,
        orbit_radius=orbit_radius_val,
        time_resolution=float(args.time_resolution) * ns,
        energy_resolution=float(args.energy_resolution),
        intrinsic_spatial_resolution=float(args.spatial_resolution) * mm,
        output_root=args.output_root,
        study_prefix=args.study_prefix,
        export_legacy_formats=not args.no_legacy,
        rng_seed=args.seed,
    )

    print(f"Обработка радионуклида {canonical_isotope} завершена. Сформировано пресетов: {len(exported)}")
    for preset_name, paths_map in exported.items():
        print(
            f"  [{preset_name}] NeXus: {paths_map['nexus']} | "
            f"DICOM: {paths_map['dicom']} | ParaView: {paths_map['paraview']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
