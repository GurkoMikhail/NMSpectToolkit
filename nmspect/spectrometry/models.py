"""Модели данных спектрометрического анализа и конфигурации энергетических окон."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PhotonComponent(IntEnum):
    """Укрупнённая 5-компонентная физическая классификация зарегистрированных квантов ОФЭКТ."""

    GEOMETRIC = 0
    SEPTAL_PENETRATION = 1
    COLLIMATOR_SCATTER = 2
    PHANTOM_SCATTER = 3
    BACKSCATTER = 4


@dataclass(frozen=True)
class EnergyWindowConfig:
    """Конфигурация энергетического окна (все энергетические границы выражены в единицах hepunits).

    Attributes:
        name: Уникальное системное имя энергетического окна.
        lower_bound: Нижняя граница энергетического окна (в базовых единицах hepunits).
        upper_bound: Верхняя граница энергетического окна (в базовых единицах hepunits).
        primary_peak: Номинальная энергия фотопика (в базовых единицах hepunits) или None для составных окон.
        description: Подробное текстовое описание окна и метода коррекции.
        is_scatter_window: Флаг принадлежности к вспомогательному окну оценки рассеяния (DEW/TEW).
        scatter_weight: Весовой коэффициент вычитания для окна рассеяния (отрицательный при вычитании).
        associated_windows: Кортеж связанных вспомогательных окон рассеяния.
    """

    name: str
    lower_bound: float
    upper_bound: float
    primary_peak: float | None = None
    description: str = ""
    is_scatter_window: bool = False
    scatter_weight: float = 0.0
    associated_windows: tuple[EnergyWindowConfig, ...] = ()

    def __post_init__(self) -> None:
        """Проверяет физическую корректность границ энергетического окна."""
        if self.lower_bound < 0.0:
            raise ValueError(f"Нижняя граница окна '{self.name}' не может быть отрицательной: {self.lower_bound}")
        if self.upper_bound <= self.lower_bound:
            raise ValueError(
                f"Верхняя граница окна '{self.name}' ({self.upper_bound}) должна быть строго больше "
                f"нижней границы ({self.lower_bound})."
            )

    @property
    def width(self) -> float:
        """Ширина энергетического окна в базовых единицах hepunits."""
        return self.upper_bound - self.lower_bound

    @property
    def center(self) -> float:
        """Центральная энергия окна в базовых единицах hepunits."""
        return 0.5 * (self.lower_bound + self.upper_bound)


@dataclass
class SpectrumResult:
    """Результат трёхуровневого спектрометрического анализа сцинтилляционных вспышек.

    Attributes:
        energy_bins: Центры энергетических бинов гистограммы (в базовых единицах hepunits).
        bin_edges: Границы энергетических бинов гистограммы (в базовых единицах hepunits).
        total_spectrum: Полный энергетический спектр всех зарегистрированных вспышек.
        primary_spectrum: Спектр первичных (нерассеянных в фантоме) фотонов.
        scatter_spectrum: Спектр рассеянных в фантоме фотонов.
        compton_continuum: Оценка гладкого комптоновского континуума.
        primary_fraction: Доля первичных фотонов в целевом окне (PF = C_prim / C_tot).
        scatter_to_primary_ratio: Отношение рассеянного излучения к первичному (SPR = C_scat / C_prim).
        compton_to_primary_ratio: Отношение комптоновского континуума к чистому фотопику (CPR).
        photopeak_efficiency: Эффективность регистрации в фотопике относительно полного числа вспышек.
        mean_scatter_order: Средняя кратность комптоновского рассеяния фотонов в окне.
        window_purity: Чистота энергетического окна (доля первичных фотонов с коррекцией фона).
        window_config: Конфигурация энергетического окна, для которого вычислены метрики.
    """

    energy_bins: NDArray[np.float64]
    bin_edges: NDArray[np.float64]
    total_spectrum: NDArray[np.float64]
    primary_spectrum: NDArray[np.float64]
    scatter_spectrum: NDArray[np.float64]
    compton_continuum: NDArray[np.float64]
    primary_fraction: float
    scatter_to_primary_ratio: float
    compton_to_primary_ratio: float
    photopeak_efficiency: float
    mean_scatter_order: float
    window_purity: float
    window_config: EnergyWindowConfig | None = None


@dataclass(frozen=True)
class EmissionBandConfig:
    """Конфигурация энергетической полосы начальной эмиссии радионуклида (в базовых единицах hepunits).

    Attributes:
        name: Уникальное системное имя полосы эмиссии.
        lower_bound: Нижняя граница начальной энергии рождения в базовых единицах `hepunits` (`MeV`).
        upper_bound: Верхняя граница начальной энергии рождения в `hepunits` (`MeV`, допускается `inf`).
        label: Читаемое русскоязычное обозначение полосы для таблиц и легенд графиков.
        include_lower: Включать ли левую границу (`>= lower_bound`), иначе строго больше (`> lower_bound`).
    """

    name: str
    lower_bound: float
    upper_bound: float
    label: str = ""
    include_lower: bool = False

    def __post_init__(self) -> None:
        """Проверяет корректность границ полосы начальной энергии."""
        if self.lower_bound < 0.0:
            raise ValueError(f"Нижняя граница полосы эмиссии '{self.name}' не может быть отрицательной.")
        if self.upper_bound <= self.lower_bound:
            raise ValueError(
                f"Верхняя граница полосы эмиссии '{self.name}' ({self.upper_bound}) "
                f"должна быть строго больше нижней ({self.lower_bound})."
            )

    def matches(self, emission_energies: NDArray[np.floating[Any]]) -> NDArray[np.bool_]:
        """Возвращает булеву маску попадания начальных энергий рождения в данную полосу."""
        energies_arr = np.asarray(emission_energies, dtype=np.float64)
        lower_mask = (
            energies_arr >= self.lower_bound if self.include_lower else energies_arr > self.lower_bound
        )
        return np.asarray(lower_mask & (energies_arr <= self.upper_bound), dtype=np.bool_)


__all__ = [
    "EmissionBandConfig",
    "EnergyWindowConfig",
    "PhotonComponent",
    "SpectrumResult",
]
