"""Модульные и интеграционные тесты генератора отчётов OOXML/OMML, графиков и CLI (nmspect.reporting)."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from nmspect.cli.track_study import main as cli_track_study_main
from nmspect.reporting import (
    DocumentFormatConverter,
    ScientificDocxBuilder,
    TrackStudyReportGenerator,
    latex_to_omml,
)
from nmspect.tracks import StreamingTrackStudyAnalyzer, TrackStudyConfig


def _build_mini_hdf5_for_report(file_path: Path) -> Path:
    """Создаёт минимальный валидный файл симуляции HDF5 для сквозного теста генератора отчётов и CLI."""
    with h5py.File(file_path, "w") as hdf_obj:
        meta = hdf_obj.create_group("metadata")
        gantry = meta.create_group("gantry")
        gantry.create_dataset("gantry_angle", data=0.0)
        gantry.create_dataset("gantry_angle_units", data="deg")
        gantry.create_dataset("orbit_radius", data=250.0)

        init = hdf_obj.create_group("initial_states")
        init.create_dataset("particle_ID", data=np.array([1, 2, 3, 4], dtype=np.int64))
        init.create_dataset(
            "emission_position",
            data=np.array(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 5.0], [0.0, 5.0, -5.0]],
                dtype=np.float64,
            ),
        )
        init.create_dataset(
            "emission_direction",
            data=np.array(
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        )
        ds_init_e = init.create_dataset("emission_energy", data=np.array([74.6, 74.6, 48.9, 290.0], dtype=np.float64))
        ds_init_e.attrs["units"] = "keV"
        init.create_dataset("emission_time", data=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))

        inter = hdf_obj.create_group("interactions")
        det1 = inter.create_group("Detector_1")
        det1.create_dataset("particle_ID", data=np.array([1, 2, 2, 3, 4, 4], dtype=np.int64))
        det1.create_dataset(
            "global_position",
            data=np.array(
                [
                    [0.0, 246.5, 0.0],
                    [12.0, 100.0, 0.0],
                    [15.0, 247.0, 0.0],
                    [-10.0, 246.0, 5.0],
                    [5.0, 262.0, 0.0],
                    [6.0, 253.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        det1.create_dataset(
            "global_direction",
            data=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        ds_det_e = det1.create_dataset(
            "energy_deposit",
            data=np.array([74.6, 4.0, 70.6, 48.9, 120.0, 75.0], dtype=np.float64),
        )
        ds_det_e.attrs["units"] = "keV"
        det1.create_dataset("distance_traveled", data=np.array([246.5, 100.0, 248.0, 246.0, 262.0, 272.0], dtype=np.float64))
        det1.create_dataset("process_code", data=np.array([3, 2, 3, 3, 2, 3], dtype=np.int32))
        det1.create_dataset("volume_id", data=np.array([6, 1, 6, 6, 8, 6], dtype=np.int32))
        det1.create_dataset("material_id", data=np.array([45, 2, 45, 45, 108, 45], dtype=np.int32))
        det1.create_dataset("Z", data=np.array([53, 8, 53, 53, 82, 53], dtype=np.int32))
        det1.create_dataset("species_code", data=np.array([0, 0, 0, 0, 0, 0], dtype=np.int32))
    return file_path


def test_latex_to_omml_and_docx_xml_validity_with_rgba_png(tmp_path: Path) -> None:
    """Проверяет транслятор LaTeX -> OMML, композитинг прозрачного RGBA PNG и строгую валидность всех XML в .docx."""
    omml_expr = latex_to_omml(
        r"\hat{P}_w = \sum_{m=1}^{10} \frac{\sqrt[3]{x_m^2 + 1}}{\left|\mathbf{r}_m - \vec{r}_0\right|}"
    )
    wrapped_xml = (
        '<root xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"{omml_expr}</root>"
    )
    parsed_omml = ET.fromstring(wrapped_xml)
    assert parsed_omml is not None

    # Создаём прозрачный RGBA PNG с цветным квадратом в центре и прозрачными полями (0, 0, 0, 0)
    rgba_arr = np.zeros((120, 160, 4), dtype=np.uint8)
    rgba_arr[30:90, 40:120] = (30, 120, 200, 255)
    transparent_png = tmp_path / "transparent_sample.png"
    Image.fromarray(rgba_arr, mode="RGBA").save(transparent_png)

    builder = ScientificDocxBuilder()
    builder.add_heading("1. Тестовый раздел", level=1)
    builder.add_paragraph(
        "Абзац с **жирным**, *курсивом*, кодом `TrackBatch` и инлайн-формулой $\\alpha + \\beta \\le 1$."
    )
    builder.add_latex_equation(r"\Omega = \bigsqcup_{m=1}^{10} C_m", equation_number=1)
    builder.add_table(
        caption="Таблица 1. Проверка",
        headers=("Категория", "Доля, %"),
        rows=(("Класс C1", "56,64"), ("Итого", "100,00")),
        column_weights=(70.0, 30.0),
        summary_rows=(-1,),
    )
    builder.add_figure(transparent_png, "Рисунок 1. Прозрачный тестовый рисунок.")
    docx_out = builder.save_docx(tmp_path / "valid_test.docx")
    assert docx_out.is_file()

    with zipfile.ZipFile(docx_out, "r") as zf:
        for member_name in zf.namelist():
            if member_name.endswith((".xml", ".rels")):
                ET.fromstring(zf.read(member_name))
        embedded_png = zf.read("word/media/image1.png")
        with Image.open(io.BytesIO(embedded_png)) as emb_img:
            corner_pixel = emb_img.getpixel((0, 0))
            assert corner_pixel == (255, 255, 255)
            assert emb_img.width < 160 and emb_img.height < 120


def test_full_report_generator_and_cli_pipeline(tmp_path: Path) -> None:
    """Проверяет сквозную генерацию всех 8 графиков, сборку отчёта .docx и работу CLI nmspect-tracks."""
    sim_dir = tmp_path / "raw_hdf"
    sim_dir.mkdir()
    _build_mini_hdf5_for_report(sim_dir / "0.0_deg.hdf")

    analyzer = StreamingTrackStudyAnalyzer(TrackStudyConfig.for_isotope("161Tb", vtk_sample_per_view=10))
    summary = analyzer.analyze_directory(sim_dir)

    generator = TrackStudyReportGenerator()
    out_docx = tmp_path / "report_out" / "study_report.docx"
    produced = generator.generate_report(summary, out_docx, export_pdf=False, export_legacy_doc=False)

    assert produced.docx_path.is_file()
    assert len(produced.figure_paths) == 8
    for fig_file in produced.figure_paths.values():
        assert fig_file.is_file()
        assert fig_file.stat().st_size > 1024

    converter = DocumentFormatConverter(timeout_duration=30.0)
    backends = converter.available_backends()
    assert isinstance(backends, tuple)
    noop_conv = converter.convert_all(produced.docx_path)
    assert noop_conv.succeeded

    # Тест CLI nmspect-tracks с экспортом VTK и сборкой отчёта .docx
    cli_out = tmp_path / "cli_results"
    exit_code = cli_track_study_main(
        [
            "--input-path",
            str(sim_dir),
            "--isotope",
            "161Tb",
            "--output-dir",
            str(cli_out),
            "--study-prefix",
            "demo",
            "--export-vtk",
            "--build-report",
        ]
    )
    assert exit_code == 0
    assert (cli_out / "demo_summary.json").is_file()
    assert (cli_out / "demo_arrays.npz").is_file()
    assert (cli_out / "demo_tracks.vtm").is_file()
    assert (cli_out / "demo_report.docx").is_file()
