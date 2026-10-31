"""CLI-точка входа (`nmspect-reconstruct`) для запуска 3D томографической реконструкции ОФЭКТ (FBP / OSEM / MLEM)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from hepunits import cm, mm

from nmspect.reconstruction import (
    FBPReconstructor,
    MLEMReconstructor,
    OSEMReconstructor,
    ReconstructionResult,
    create_uniform_attenuation_map,
    estimate_attenuation_map_from_projections,
    get_water_linear_attenuation,
    load_attenuation_volume,
    load_projections,
    save_slices_preview_png,
    save_volume_as_npy,
    save_volume_as_vti,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для 3D томографической реконструкции."""
    parser = argparse.ArgumentParser(
        prog="nmspect-reconstruct",
        description="3D томографическая реконструкция проекционных данных ОФЭКТ методами FBP, OSEM и MLEM.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Путь к входному файлу проекций (.nxs, .dcm, .npy) или директории со срезами (.spr/.dat).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fbp", "osem", "mlem"],
        default="osem",
        help="Алгоритм реконструкции: 'fbp', 'osem' или 'mlem' (по умолчанию: osem).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reconstruction",
        help="Директория для сохранения реконструированного 3D-объёма (по умолчанию: results/reconstruction).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Базовое имя выходных файлов (по умолчанию формируется из имени входного файла и метода).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Размер вокселя в мм (по умолчанию равен размеру пикселя проекций).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Количество итераций для OSEM / MLEM (по умолчанию: 8).",
    )
    parser.add_argument(
        "--subsets",
        type=int,
        default=8,
        help="Количество упорядоченных подмножеств для OSEM (по умолчанию: 8).",
    )
    parser.add_argument(
        "--filter",
        dest="filter_name",
        type=str,
        default="shepp-logan",
        choices=["ramp", "ram-lak", "shepp-logan", "cosine", "hamming", "hann", "butterworth"],
        help="Частотный фильтр для FBP (по умолчанию: shepp-logan).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.85,
        help="Частота среза фильтра FBP в долях частоты Найквиста (по умолчанию: 0.85).",
    )
    parser.add_argument(
        "--attenuation-mode",
        type=str,
        choices=["none", "uniform", "auto-contour", "file"],
        default="none",
        help="Режим коррекции затухания фотонов (по умолчанию: none).",
    )
    parser.add_argument(
        "--attenuation-file",
        type=str,
        default=None,
        help="Путь к файлу внешней карты затухания (.npy или .dat) при --attenuation-mode=file.",
    )
    parser.add_argument(
        "--isotope",
        type=str,
        default="161Tb",
        help="Изотоп для выбора табличного коэффициента ослабления в воде ('161Tb' или '99mTc').",
    )
    parser.add_argument(
        "--linear-attenuation",
        type=float,
        default=None,
        help="Явный линейный коэффициент ослабления в 1/см (переопределяет табличное значение изотопа).",
    )
    parser.add_argument(
        "--phantom-radius",
        type=float,
        default=None,
        help="Радиус однородного цилиндрического фантома в мм при --attenuation-mode=uniform.",
    )
    parser.add_argument(
        "--psf-fwhm",
        type=float,
        default=None,
        help="Ширина функции размытия точки детектора (PSF FWHM) в мм при проецировании.",
    )
    parser.add_argument(
        "--post-smoothing",
        type=float,
        default=None,
        help="Пост-реконструкционное 3D гауссово сглаживание (FWHM) в мм.",
    )
    parser.add_argument(
        "--prior",
        type=str,
        choices=["none", "quadratic", "tv"],
        default="none",
        help="Тип регуляризации OSL (априорного штрафа) в OSEM/MLEM.",
    )
    parser.add_argument(
        "--prior-beta",
        type=float,
        default=0.02,
        help="Коэффициент регуляризации beta при включённом --prior.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI-команды `nmspect-reconstruct`."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    projections_raw, angles, loaded_pixel_size = load_projections(input_path)
    projections = np.asarray(projections_raw, dtype=np.float64)
    voxel_size = float(args.voxel_size) * mm if args.voxel_size is not None else float(loaded_pixel_size)

    _, transaxial_size, num_slices = projections.shape
    volume_shape = (int(transaxial_size), int(transaxial_size), int(num_slices))

    custom_mu = float(args.linear_attenuation) / cm if args.linear_attenuation is not None else None
    water_mu = get_water_linear_attenuation(isotope=args.isotope, custom_mu=custom_mu)

    attenuation_map: np.ndarray | None = None
    if args.attenuation_mode == "uniform":
        radius_val = float(args.phantom_radius) * mm if args.phantom_radius is not None else None
        attenuation_map = create_uniform_attenuation_map(
            shape=volume_shape,
            linear_attenuation=water_mu,
            voxel_size=voxel_size,
            phantom_radius=radius_val,
        )
    elif args.attenuation_mode == "auto-contour":
        attenuation_map = estimate_attenuation_map_from_projections(
            projections=projections,
            angles=angles,
            linear_attenuation=water_mu,
            voxel_size=voxel_size,
        )
    elif args.attenuation_mode == "file":
        if args.attenuation_file is None:
            raise ValueError("При --attenuation-mode=file необходимо указать --attenuation-file.")
        attenuation_map = load_attenuation_volume(args.attenuation_file, shape=volume_shape)

    psf_val = float(args.psf_fwhm) * mm if args.psf_fwhm is not None else None
    smooth_val = float(args.post_smoothing) * mm if args.post_smoothing is not None else None
    prior_type = None if args.prior == "none" else args.prior

    result: ReconstructionResult
    if args.method == "fbp":
        fbp_reconstructor = FBPReconstructor(
            filter_name=args.filter_name,
            cutoff_frequency=float(args.cutoff),
            voxel_size=voxel_size,
            attenuation_map=attenuation_map,
            post_smoothing_fwhm=smooth_val,
        )
        result = fbp_reconstructor.reconstruct_with_metadata(projections, angles=angles)
    elif args.method == "mlem":
        mlem_reconstructor = MLEMReconstructor(
            num_iterations=int(args.iterations),
            voxel_size=voxel_size,
            attenuation_map=attenuation_map,
            post_smoothing_fwhm=smooth_val,
            psf_fwhm=psf_val,
            prior_type=prior_type,
            prior_beta=float(args.prior_beta) if prior_type is not None else 0.0,
        )
        result = mlem_reconstructor.reconstruct_with_metadata(projections, angles=angles)
    else:
        osem_reconstructor = OSEMReconstructor(
            num_iterations=int(args.iterations),
            num_subsets=int(args.subsets),
            voxel_size=voxel_size,
            attenuation_map=attenuation_map,
            post_smoothing_fwhm=smooth_val,
            psf_fwhm=psf_val,
            prior_type=prior_type,
            prior_beta=float(args.prior_beta) if prior_type is not None else 0.0,
        )
        result = osem_reconstructor.reconstruct_with_metadata(projections, angles=angles)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.output_name if args.output_name is not None else f"{input_path.stem}_{args.method}"

    npy_path = save_volume_as_npy(result.volume, out_dir / f"{base_name}.npy")
    vti_path = save_volume_as_vti(
        result.volume,
        out_dir / f"{base_name}.vti",
        voxel_size=voxel_size,
        attenuation_map=attenuation_map,
    )
    png_path = save_slices_preview_png(result.volume, out_dir / f"{base_name}_slices.png")

    meta_path = out_dir / f"{base_name}_metadata.json"
    serializable_meta = {
        "method": result.method,
        "voxel_size": float(result.voxel_size / mm),
        "voxel_size_units": "mm",
        "volume_shape": list(result.volume.shape),
        "attenuation_corrected": bool(result.attenuation_corrected),
        "input_path": str(input_path),
    }
    meta_path.write_text(json.dumps(serializable_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Реконструкция ({result.method}) успешно завершена:")
    print(f"  NPY : {npy_path}")
    print(f"  VTI : {vti_path}")
    print(f"  PNG : {png_path}")
    print(f"  JSON: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
