"""Пакет командных интерфейсов (CLI) NMSpectToolkit."""

from __future__ import annotations

from nmspect.cli.inspect_nexus import main as main_nexus
from nmspect.cli.process import main as main_process
from nmspect.cli.reconstruct import main as main_reconstruct
from nmspect.cli.viewers import main_spectrum, main_vtk

__all__ = [
    "main_nexus",
    "main_process",
    "main_reconstruct",
    "main_spectrum",
    "main_vtk",
]
