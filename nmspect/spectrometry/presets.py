"""Реестр предустановленных конфигураций энергетических окон ОФЭКТ для радионуклидов 161Tb и 99mTc."""

from __future__ import annotations

from typing import Any

import numpy as np
from hepunits import keV
from numpy.typing import NDArray

from nmspect.spectrometry.models import EmissionBandConfig, EnergyWindowConfig
from nmspect.spectrometry.scatter import calculate_dew_coefficient, calculate_tew_coefficients

# ---------------------------------------------------------------------------
# Физические константы основных фотопиков (в базовых единицах hepunits)
# ---------------------------------------------------------------------------
TB161_PEAK_49: float = 48.9 * keV
TB161_PEAK_75: float = 74.6 * keV
TC99M_PEAK_140: float = 140.5 * keV

# ---------------------------------------------------------------------------
# Конфигурации энергетических окон для 161Tb (тербий-161)
# ---------------------------------------------------------------------------
TB161_PEAK_49_WINDOW = EnergyWindowConfig(
    name="peak_49",
    lower_bound=44.0 * keV,
    upper_bound=53.8 * keV,
    primary_peak=TB161_PEAK_49,
    description="Основное окно 48.9 кэВ (±10%: 44.0–53.8 кэВ), без коррекции рассеяния",
)

TB161_PEAK_75_WINDOW = EnergyWindowConfig(
    name="peak_75",
    lower_bound=67.1 * keV,
    upper_bound=82.1 * keV,
    primary_peak=TB161_PEAK_75,
    description="Основное окно 74.6 кэВ (±10%: 67.1–82.1 кэВ), без коррекции рассеяния",
)

# DEW для пика 48.9 кэВ
_tb161_dew_49_weight = calculate_dew_coefficient(
    main_lower=44.0 * keV,
    main_upper=53.8 * keV,
    scatter_lower=35.0 * keV,
    scatter_upper=43.0 * keV,
    k_factor=0.5,
)
TB161_SCATTER_49_DEW = EnergyWindowConfig(
    name="scatter_49_dew",
    lower_bound=35.0 * keV,
    upper_bound=43.0 * keV,
    primary_peak=TB161_PEAK_49,
    description="Нижнее окно рассеяния DEW для пика 48.9 кэВ (35.0–43.0 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_dew_49_weight,
)
TB161_PEAK_49_DEW_WINDOW = EnergyWindowConfig(
    name="peak_49_dew",
    lower_bound=44.0 * keV,
    upper_bound=53.8 * keV,
    primary_peak=TB161_PEAK_49,
    description="Окно 48.9 кэВ (44.0–53.8 кэВ) с коррекцией рассеяния DEW (35.0–43.0 кэВ)",
    associated_windows=(TB161_SCATTER_49_DEW,),
)

# TEW для пика 48.9 кэВ
_tb161_tew_49_low_w, _tb161_tew_49_high_w = calculate_tew_coefficients(
    main_lower=44.0 * keV,
    main_upper=53.8 * keV,
    low_scatter_lower=40.0 * keV,
    low_scatter_upper=44.0 * keV,
    high_scatter_lower=53.8 * keV,
    high_scatter_upper=57.8 * keV,
)
TB161_SCATTER_49_LOW = EnergyWindowConfig(
    name="scatter_49_low",
    lower_bound=40.0 * keV,
    upper_bound=44.0 * keV,
    primary_peak=TB161_PEAK_49,
    description="Нижнее окно рассеяния TEW для пика 48.9 кэВ (40.0–44.0 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_tew_49_low_w,
)
TB161_SCATTER_49_HIGH = EnergyWindowConfig(
    name="scatter_49_high",
    lower_bound=53.8 * keV,
    upper_bound=57.8 * keV,
    primary_peak=TB161_PEAK_49,
    description="Верхнее окно рассеяния TEW для пика 48.9 кэВ (53.8–57.8 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_tew_49_high_w,
)
TB161_PEAK_49_TEW_WINDOW = EnergyWindowConfig(
    name="peak_49_tew",
    lower_bound=44.0 * keV,
    upper_bound=53.8 * keV,
    primary_peak=TB161_PEAK_49,
    description="Окно 48.9 кэВ (44.0–53.8 кэВ) с коррекцией рассеяния TEW (40.0–44.0 и 53.8–57.8 кэВ)",
    associated_windows=(TB161_SCATTER_49_LOW, TB161_SCATTER_49_HIGH),
)

# DEW для пика 74.6 кэВ
_tb161_dew_75_weight = calculate_dew_coefficient(
    main_lower=67.1 * keV,
    main_upper=82.1 * keV,
    scatter_lower=57.0 * keV,
    scatter_upper=66.0 * keV,
    k_factor=0.5,
)
TB161_SCATTER_75_DEW = EnergyWindowConfig(
    name="scatter_75_dew",
    lower_bound=57.0 * keV,
    upper_bound=66.0 * keV,
    primary_peak=TB161_PEAK_75,
    description="Нижнее окно рассеяния DEW для пика 74.6 кэВ (57.0–66.0 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_dew_75_weight,
)
TB161_PEAK_75_DEW_WINDOW = EnergyWindowConfig(
    name="peak_75_dew",
    lower_bound=67.1 * keV,
    upper_bound=82.1 * keV,
    primary_peak=TB161_PEAK_75,
    description="Окно 74.6 кэВ (67.1–82.1 кэВ) с коррекцией рассеяния DEW (57.0–66.0 кэВ)",
    associated_windows=(TB161_SCATTER_75_DEW,),
)

# TEW для пика 74.6 кэВ
_tb161_tew_75_low_w, _tb161_tew_75_high_w = calculate_tew_coefficients(
    main_lower=67.1 * keV,
    main_upper=82.1 * keV,
    low_scatter_lower=61.1 * keV,
    low_scatter_upper=67.1 * keV,
    high_scatter_lower=82.1 * keV,
    high_scatter_upper=88.1 * keV,
)
TB161_SCATTER_75_LOW = EnergyWindowConfig(
    name="scatter_75_low",
    lower_bound=61.1 * keV,
    upper_bound=67.1 * keV,
    primary_peak=TB161_PEAK_75,
    description="Нижнее окно рассеяния TEW для пика 74.6 кэВ (61.1–67.1 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_tew_75_low_w,
)
TB161_SCATTER_75_HIGH = EnergyWindowConfig(
    name="scatter_75_high",
    lower_bound=82.1 * keV,
    upper_bound=88.1 * keV,
    primary_peak=TB161_PEAK_75,
    description="Верхнее окно рассеяния TEW для пика 74.6 кэВ (82.1–88.1 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_tew_75_high_w,
)
TB161_PEAK_75_TEW_WINDOW = EnergyWindowConfig(
    name="peak_75_tew",
    lower_bound=67.1 * keV,
    upper_bound=82.1 * keV,
    primary_peak=TB161_PEAK_75,
    description="Окно 74.6 кэВ (67.1–82.1 кэВ) с коррекцией рассеяния TEW (61.1–67.1 и 82.1–88.1 кэВ)",
    associated_windows=(TB161_SCATTER_75_LOW, TB161_SCATTER_75_HIGH),
)

# Симметричные 10%-ные подокна TEW и калиброванный двухпиковый протокол 161Tb (протокол NEMA Symbia)
_tb161_sym_49_low_w, _tb161_sym_49_high_w = calculate_tew_coefficients(
    main_lower=44.0 * keV,
    main_upper=53.8 * keV,
    low_scatter_lower=39.1 * keV,
    low_scatter_upper=44.0 * keV,
    high_scatter_lower=53.8 * keV,
    high_scatter_upper=58.7 * keV,
)
TB161_SCATTER_49_SYM_LOW = EnergyWindowConfig(
    name="scatter_49_sym_low",
    lower_bound=39.1 * keV,
    upper_bound=44.0 * keV,
    primary_peak=TB161_PEAK_49,
    description="Нижнее 10%-ное окно TEW для пика 48.9 кэВ (39.1–44.0 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_sym_49_low_w,
)
TB161_SCATTER_49_SYM_HIGH = EnergyWindowConfig(
    name="scatter_49_sym_high",
    lower_bound=53.8 * keV,
    upper_bound=58.7 * keV,
    primary_peak=TB161_PEAK_49,
    description="Верхнее 10%-ное окно TEW для пика 48.9 кэВ (53.8–58.7 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_sym_49_high_w,
)
TB161_PEAK_49_TEW_SYM_WINDOW = EnergyWindowConfig(
    name="peak_49_tew_sym",
    lower_bound=44.0 * keV,
    upper_bound=53.8 * keV,
    primary_peak=TB161_PEAK_49,
    description="Окно 48.9 кэВ (44.0–53.8 кэВ) с симметричным TEW (39.1–44.0 и 53.8–58.7 кэВ)",
    associated_windows=(TB161_SCATTER_49_SYM_LOW, TB161_SCATTER_49_SYM_HIGH),
)

_tb161_sym_75_low_w, _tb161_sym_75_high_w = calculate_tew_coefficients(
    main_lower=67.1 * keV,
    main_upper=82.0 * keV,
    low_scatter_lower=59.7 * keV,
    low_scatter_upper=67.1 * keV,
    high_scatter_lower=82.0 * keV,
    high_scatter_upper=89.5 * keV,
)
TB161_SCATTER_75_SYM_LOW = EnergyWindowConfig(
    name="scatter_75_sym_low",
    lower_bound=59.7 * keV,
    upper_bound=67.1 * keV,
    primary_peak=TB161_PEAK_75,
    description="Нижнее 10%-ное окно TEW для пика 74.6 кэВ (59.7–67.1 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_sym_75_low_w,
)
TB161_SCATTER_75_SYM_HIGH = EnergyWindowConfig(
    name="scatter_75_sym_high",
    lower_bound=82.0 * keV,
    upper_bound=89.5 * keV,
    primary_peak=TB161_PEAK_75,
    description="Верхнее 10%-ное окно TEW для пика 74.6 кэВ (82.0–89.5 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tb161_sym_75_high_w,
)
TB161_PEAK_75_TEW_SYM_WINDOW = EnergyWindowConfig(
    name="peak_75_tew_sym",
    lower_bound=67.1 * keV,
    upper_bound=82.0 * keV,
    primary_peak=TB161_PEAK_75,
    description="Окно 74.6 кэВ (67.1–82.0 кэВ) с симметричным TEW (59.7–67.1 и 82.0–89.5 кэВ)",
    associated_windows=(TB161_SCATTER_75_SYM_LOW, TB161_SCATTER_75_SYM_HIGH),
)

TB161_ENERGY_WINDOW_PRESETS: dict[str, tuple[EnergyWindowConfig, ...]] = {
    "peak49": (TB161_PEAK_49_WINDOW,),
    "peak75": (TB161_PEAK_75_WINDOW,),
    "dual_peak": (TB161_PEAK_49_WINDOW, TB161_PEAK_75_WINDOW),
    "peak49_dew": (TB161_PEAK_49_DEW_WINDOW,),
    "peak49_tew": (TB161_PEAK_49_TEW_WINDOW,),
    "peak75_dew": (TB161_PEAK_75_DEW_WINDOW,),
    "peak75_tew": (TB161_PEAK_75_TEW_WINDOW,),
    "peak49_tew_sym": (TB161_PEAK_49_TEW_SYM_WINDOW,),
    "peak75_tew_sym": (TB161_PEAK_75_TEW_SYM_WINDOW,),
    "dual_peak_tew_sym": (TB161_PEAK_49_TEW_SYM_WINDOW, TB161_PEAK_75_TEW_SYM_WINDOW),
}

# ---------------------------------------------------------------------------
# Конфигурации энергетических окон для 99mTc (технеций-99м)
# ---------------------------------------------------------------------------
TC99M_PEAK_140_WINDOW = EnergyWindowConfig(
    name="peak_140",
    lower_bound=126.5 * keV,
    upper_bound=154.6 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Стандартное клиническое окно 140.5 кэВ (20%, ±10%: 126.5–154.6 кэВ), без коррекции рассеяния",
)

TC99M_PEAK_140_15PCT_WINDOW = EnergyWindowConfig(
    name="peak_140_15pct",
    lower_bound=130.0 * keV,
    upper_bound=151.0 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Узкое клиническое окно 140.5 кэВ (15%, ±7.5%: 130.0–151.0 кэВ), повышенный контраст",
)

# DEW (стандартный k = 0.5) для 99mTc
_tc99m_dew_weight = calculate_dew_coefficient(
    main_lower=126.5 * keV,
    main_upper=154.6 * keV,
    scatter_lower=106.0 * keV,
    scatter_upper=125.0 * keV,
    k_factor=0.5,
)
TC99M_SCATTER_140_DEW = EnergyWindowConfig(
    name="scatter_140_dew",
    lower_bound=106.0 * keV,
    upper_bound=125.0 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Нижнее окно рассеяния DEW для 140.5 кэВ (106.0–125.0 кэВ, k=0.5)",
    is_scatter_window=True,
    scatter_weight=_tc99m_dew_weight,
)
TC99M_PEAK_140_DEW_WINDOW = EnergyWindowConfig(
    name="peak_140_dew",
    lower_bound=126.5 * keV,
    upper_bound=154.6 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Окно 140.5 кэВ (126.5–154.6 кэВ) с коррекцией рассеяния DEW (106.0–125.0 кэВ, k=0.5)",
    associated_windows=(TC99M_SCATTER_140_DEW,),
)

# DEW по Jaszczak (классическое окно Комптона 92–125 кэВ, k = 0.5)
_tc99m_jaszczak_weight = calculate_dew_coefficient(
    main_lower=126.5 * keV,
    main_upper=154.6 * keV,
    scatter_lower=92.0 * keV,
    scatter_upper=125.0 * keV,
    k_factor=0.5,
)
TC99M_SCATTER_140_JASZCZAK = EnergyWindowConfig(
    name="scatter_140_jaszczak",
    lower_bound=92.0 * keV,
    upper_bound=125.0 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Широкое комптоновское окно Jaszczak для 140.5 кэВ (92.0–125.0 кэВ, k=0.5)",
    is_scatter_window=True,
    scatter_weight=_tc99m_jaszczak_weight,
)
TC99M_PEAK_140_DEW_JASZCZAK_WINDOW = EnergyWindowConfig(
    name="peak_140_dew_jaszczak",
    lower_bound=126.5 * keV,
    upper_bound=154.6 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Окно 140.5 кэВ (126.5–154.6 кэВ) с коррекцией DEW по Jaszczak (92.0–125.0 кэВ)",
    associated_windows=(TC99M_SCATTER_140_JASZCZAK,),
)

# TEW (Ogawa) для 99mTc
_tc99m_tew_low_w, _tc99m_tew_high_w = calculate_tew_coefficients(
    main_lower=126.5 * keV,
    main_upper=154.6 * keV,
    low_scatter_lower=120.5 * keV,
    low_scatter_upper=126.5 * keV,
    high_scatter_lower=154.6 * keV,
    high_scatter_upper=160.6 * keV,
)
TC99M_SCATTER_140_LOW = EnergyWindowConfig(
    name="scatter_140_low",
    lower_bound=120.5 * keV,
    upper_bound=126.5 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Нижнее окно рассеяния TEW для 140.5 кэВ (120.5–126.5 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tc99m_tew_low_w,
)
TC99M_SCATTER_140_HIGH = EnergyWindowConfig(
    name="scatter_140_high",
    lower_bound=154.6 * keV,
    upper_bound=160.6 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Верхнее окно рассеяния TEW для 140.5 кэВ (154.6–160.6 кэВ)",
    is_scatter_window=True,
    scatter_weight=_tc99m_tew_high_w,
)
TC99M_PEAK_140_TEW_WINDOW = EnergyWindowConfig(
    name="peak_140_tew",
    lower_bound=126.5 * keV,
    upper_bound=154.6 * keV,
    primary_peak=TC99M_PEAK_140,
    description="Окно 140.5 кэВ (126.5–154.6 кэВ) с коррекцией рассеяния TEW (120.5–126.5 и 154.6–160.6 кэВ)",
    associated_windows=(TC99M_SCATTER_140_LOW, TC99M_SCATTER_140_HIGH),
)

TC99M_ENERGY_WINDOW_PRESETS: dict[str, tuple[EnergyWindowConfig, ...]] = {
    "peak140": (TC99M_PEAK_140_WINDOW,),
    "peak140_15pct": (TC99M_PEAK_140_15PCT_WINDOW,),
    "peak140_dew": (TC99M_PEAK_140_DEW_WINDOW,),
    "peak140_dew_jaszczak": (TC99M_PEAK_140_DEW_JASZCZAK_WINDOW,),
    "peak140_tew": (TC99M_PEAK_140_TEW_WINDOW,),
}

# ---------------------------------------------------------------------------
# Единый реестр пресетов по радионуклидам
# ---------------------------------------------------------------------------
ISOTOPE_PRESET_REGISTRY: dict[str, dict[str, tuple[EnergyWindowConfig, ...]]] = {
    "161Tb": TB161_ENERGY_WINDOW_PRESETS,
    "99mTc": TC99M_ENERGY_WINDOW_PRESETS,
}

_ISOTOPE_ALIASES: dict[str, str] = {
    "161tb": "161Tb",
    "tb161": "161Tb",
    "tb-161": "161Tb",
    "99mtc": "99mTc",
    "tc99m": "99mTc",
    "tc-99m": "99mTc",
}


def normalize_isotope_name(isotope: str) -> str:
    """Приводит строковое обозначение радионуклида к каноническому виду ('161Tb' или '99mTc')."""
    normalized_key = isotope.strip().lower()
    if normalized_key not in _ISOTOPE_ALIASES:
        supported = ", ".join(sorted(ISOTOPE_PRESET_REGISTRY.keys()))
        raise ValueError(f"Неизвестный радионуклид '{isotope}'. Поддерживаемые радионуклиды: {supported}")
    return _ISOTOPE_ALIASES[normalized_key]


def get_energy_window_preset(
    preset_key: str,
    isotope: str | None = None,
) -> tuple[EnergyWindowConfig, ...]:
    """Возвращает кортеж конфигураций энергетических окон по имени пресета и (опционально) радионуклиду.

    Args:
        preset_key: Ключ пресета (например, 'peak75_tew', 'dual_peak', 'peak140_dew').
        isotope: Обозначение радионуклида ('161Tb', 'tb161', '99mTc', 'tc99m') или None для сквозного поиска.

    Returns:
        Кортеж объектов EnergyWindowConfig.
    """
    normalized_preset = preset_key.strip().lower()

    if isotope is not None:
        canonical_isotope = normalize_isotope_name(isotope)
        presets_map = ISOTOPE_PRESET_REGISTRY[canonical_isotope]
        if normalized_preset not in presets_map:
            available = ", ".join(presets_map.keys())
            raise KeyError(
                f"Неизвестный пресет '{preset_key}' для радионуклида {canonical_isotope}. "
                f"Доступные пресеты: {available}"
            )
        return presets_map[normalized_preset]

    for presets_map in ISOTOPE_PRESET_REGISTRY.values():
        if normalized_preset in presets_map:
            return presets_map[normalized_preset]

    all_available = ", ".join(list_available_presets())
    raise KeyError(f"Неизвестный пресет энергетического окна '{preset_key}'. Доступные пресеты: {all_available}")


def list_available_presets(isotope: str | None = None) -> list[str]:
    """Возвращает список ключей доступных пресетов энергетических окон."""
    if isotope is not None:
        canonical_isotope = normalize_isotope_name(isotope)
        return list(ISOTOPE_PRESET_REGISTRY[canonical_isotope].keys())

    combined: list[str] = []
    for presets_map in ISOTOPE_PRESET_REGISTRY.values():
        combined.extend(presets_map.keys())
    return combined


def format_presets_summary(isotope: str | None = None) -> str:
    """Формирует табличную сводку доступных пресетов энергетических окон на русском языке."""
    target_isotopes = (
        [normalize_isotope_name(isotope)] if isotope is not None else list(ISOTOPE_PRESET_REGISTRY.keys())
    )
    lines: list[str] = []
    for iso_name in target_isotopes:
        lines.append(f"Пресеты энергетических окон для {iso_name}:")
        for key, windows in ISOTOPE_PRESET_REGISTRY[iso_name].items():
            win_descriptions = "; ".join(
                f"{win.name} [{win.lower_bound / keV:.1f}–{win.upper_bound / keV:.1f} кэВ]"
                for win in windows
            )
            lines.append(f"  - {key:<22}: {win_descriptions} — {windows[0].description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Пресеты полос начальной эмиссии радионуклидов и автоматическая кластеризация
# ---------------------------------------------------------------------------
TB161_EMISSION_BANDS: tuple[EmissionBandConfig, ...] = (
    EmissionBandConfig(
        name="band_26",
        lower_bound=0.0 * keV,
        upper_bound=35.0 * keV,
        label="25,7 кэВ",
        include_lower=True,
    ),
    EmissionBandConfig(
        name="band_49_multiplet",
        lower_bound=35.0 * keV,
        upper_bound=65.0 * keV,
        label="Мультиплет 45–59 кэВ",
        include_lower=False,
    ),
    EmissionBandConfig(
        name="band_75_multiplet",
        lower_bound=65.0 * keV,
        upper_bound=82.0 * keV,
        label="Дублет 74,6–77,4 кэВ",
        include_lower=False,
    ),
    EmissionBandConfig(
        name="band_88_106",
        lower_bound=82.0 * keV,
        upper_bound=150.0 * keV,
        label="Линии 87,9–106,1 кэВ",
        include_lower=False,
    ),
    EmissionBandConfig(
        name="band_high_gamma",
        lower_bound=150.0 * keV,
        upper_bound=float("inf"),
        label="Высокоэнергетические γ (>150 кэВ)",
        include_lower=False,
    ),
)

TC99M_EMISSION_BANDS: tuple[EmissionBandConfig, ...] = (
    EmissionBandConfig(
        name="band_low_xray",
        lower_bound=0.0 * keV,
        upper_bound=100.0 * keV,
        label="Рентгеновские линии (<100 кэВ)",
        include_lower=True,
    ),
    EmissionBandConfig(
        name="band_140",
        lower_bound=100.0 * keV,
        upper_bound=160.0 * keV,
        label="Основной γ-пик 140,5 кэВ",
        include_lower=False,
    ),
    EmissionBandConfig(
        name="band_high",
        lower_bound=160.0 * keV,
        upper_bound=float("inf"),
        label="Старшие переходы (>160 кэВ)",
        include_lower=False,
    ),
)

ISOTOPE_EMISSION_BANDS_REGISTRY: dict[str, tuple[EmissionBandConfig, ...]] = {
    "161Tb": TB161_EMISSION_BANDS,
    "99mTc": TC99M_EMISSION_BANDS,
}


def get_emission_bands_preset(isotope: str) -> tuple[EmissionBandConfig, ...]:
    """Возвращает дизъюнктный набор полос начальной энергии эмиссии для заданного радионуклида."""
    canonical_name = normalize_isotope_name(isotope)
    return ISOTOPE_EMISSION_BANDS_REGISTRY[canonical_name]


def infer_emission_bands_from_energies(
    emission_energies: NDArray[np.floating[Any]],
    cluster_gap: float = 12.0 * keV,
) -> tuple[EmissionBandConfig, ...]:
    """Автоматически строит полное дизъюнктное разбиение полос эмиссии `[0, +inf)` по массиву энергий рождения.

    Группирует близкие дискретные линии эмиссии, отстоящие не более чем на `cluster_gap`,
    и ставит границы полос посередине между соседними мультиплетами.
    """
    if cluster_gap <= 0.0:
        raise ValueError(f"Параметр cluster_gap должен быть строго положительным: {cluster_gap}.")

    energies_arr = np.asarray(emission_energies, dtype=np.float64).ravel()
    positive_vals = energies_arr[np.isfinite(energies_arr) & (energies_arr >= 0.0)]
    if positive_vals.size == 0:
        return (
            EmissionBandConfig(
                name="band_all",
                lower_bound=0.0 * keV,
                upper_bound=float("inf"),
                label="Весь спектр эмиссии",
                include_lower=True,
            ),
        )

    rounded_kevs = np.unique(np.round(positive_vals / float(keV), decimals=2)) * float(keV)
    clusters: list[list[float]] = [[float(rounded_kevs[0])]]
    for line_val in rounded_kevs[1:]:
        if float(line_val) - clusters[-1][-1] <= float(cluster_gap):
            clusters[-1].append(float(line_val))
        else:
            clusters.append([float(line_val)])

    if len(clusters) == 1:
        mean_line_value = float(np.mean(clusters[0])) / float(keV)
        return (
            EmissionBandConfig(
                name=f"band_{round(mean_line_value)}",
                lower_bound=0.0 * keV,
                upper_bound=float("inf"),
                label=f"{mean_line_value:.1f} кэВ",
                include_lower=True,
            ),
        )

    bands: list[EmissionBandConfig] = []
    num_clusters = len(clusters)
    prev_upper = 0.0
    for cluster_idx, cluster_lines in enumerate(clusters):
        is_first = cluster_idx == 0
        is_last = cluster_idx == num_clusters - 1
        next_upper = (
            float("inf")
            if is_last
            else 0.5 * (cluster_lines[-1] + clusters[cluster_idx + 1][0])
        )
        min_line_value = cluster_lines[0] / float(keV)
        max_line_value = cluster_lines[-1] / float(keV)
        if len(cluster_lines) == 1:
            band_label = f"{min_line_value:.1f} кэВ"
            band_name = f"band_{round(min_line_value)}"
        else:
            band_label = f"{min_line_value:.1f}–{max_line_value:.1f} кэВ"
            band_name = f"band_{round(min_line_value)}_{round(max_line_value)}"

        bands.append(
            EmissionBandConfig(
                name=band_name,
                lower_bound=prev_upper,
                upper_bound=next_upper,
                label=band_label,
                include_lower=is_first,
            )
        )
        prev_upper = next_upper

    return tuple(bands)
