"""CLI-точка входа (`nmspect-nexus`) для инспекции дерева и построения графиков файлов HDF5 NeXus (.nxs)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from nmspect.visualization.nexus_inspector import (
    format_nexus_tree,
    inspect_nexus_file,
    plot_nexus_projections,
    plot_nexus_spectrum,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки для инспекции файлов NeXus."""
    parser = argparse.ArgumentParser(
        prog="nmspect-nexus",
        description="Инспекция иерархии, метрик качества и визуализация спектров/проекций файлов HDF5 NeXus (.nxs).",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Путь к исследуемому файлу NeXus (.nxs).",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Вывести полное иерархическое дерево групп, датасетов и атрибутов файла.",
    )
    parser.add_argument(
        "--plot-spectrum",
        type=str,
        default=None,
        help="Сохранить график трёхуровневого спектра в указанный файл (.png / .pdf).",
    )
    parser.add_argument(
        "--plot-projections",
        type=str,
        default=None,
        help="Сохранить сетку планарных проекций в указанный файл (.png / .pdf).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Открыть интерактивные окна графиков Matplotlib.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI-команды `nmspect-nexus`."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.tree:
        print(format_nexus_tree(args.file_path))
    else:
        summary = inspect_nexus_file(args.file_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.plot_spectrum is not None or args.show:
        plot_nexus_spectrum(
            args.file_path,
            output_image_path=args.plot_spectrum,
            show_interactive=bool(args.show),
        )

    if args.plot_projections is not None:
        plot_nexus_projections(
            args.file_path,
            output_image_path=args.plot_projections,
            show_interactive=bool(args.show),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
