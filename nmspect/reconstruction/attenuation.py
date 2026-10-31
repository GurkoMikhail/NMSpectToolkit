"""
Модуль построения карт ослабления гамма-излучения и коррекции затухания по Чангу.

Реализует:
- справочник линейных коэффициентов ослабления гамма-квантов в воде (мягких тканях)
  для медицинских радионуклидов в системе единиц `hepunits`;
- аналитическое построение однородных цилиндрических 2D/3D карт затухания;
- автоматическую оценку контура объекта (Visual Hull) по проекционным теням;
- расчёт поправочной матрицы затухания по методу Чанга и коррекцию первого порядка.

Все физические величины передаются и обрабатываются строго в системе единиц `hepunits`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from hepunits import cm, mm
from numpy.typing import NDArray

from nmspect.reconstruction.projector import (
    TomographicProjector,
    normalize_projection_angles,
)

# Справочник линейных коэффициентов ослабления гамма-излучения в воде (мягких тканях)
# для основных радионуклидов ядерной медицины в системе единиц hepunits (1 / длина)
WATER_LINEAR_ATTENUATION_BY_ISOTOPE: dict[str, float] = {
    "161Tb": 0.188 / cm,  # Фотопик 74.57 кэВ (вода, NIST XCOM)
    "99mTc": 0.154 / cm,  # Фотопик 140.51 кэВ
    "123I": 0.147 / cm,   # Фотопик 159.0 кэВ
    "177Lu": 0.135 / cm,  # Фотопик 208.37 кэВ
    "111In": 0.140 / cm,  # Фотопики 171.3 / 245.4 кэВ
    "131I": 0.110 / cm,   # Фотопик 364.49 кэВ
}


def get_water_linear_attenuation(
    isotope: str | None = "161Tb",
    custom_mu: float | None = None,
) -> float:
    """
    Возвращает линейный коэффициент ослабления в воде для указанного радионуклида
    (либо пользовательское значение `custom_mu`) в единицах `hepunits`.

    :param isotope: Обозначение изотопа (например, `'161Tb'`, `'99mTc'`, `'177Lu'`).
    :param custom_mu: Опциональное пользовательское значение линейного коэффициента ослабления в `hepunits`.
    :return: Линейный коэффициент ослабления в системе `hepunits`.
    """
    if custom_mu is not None:
        custom_val = float(custom_mu)
        if custom_val < 0.0:
            raise ValueError(
                f"Линейный коэффициент ослабления не может быть отрицательным: {custom_val}."
            )
        return custom_val

    target_isotope = isotope if isotope is not None else "161Tb"
    cleaned = target_isotope.strip()
    if cleaned in WATER_LINEAR_ATTENUATION_BY_ISOTOPE:
        return WATER_LINEAR_ATTENUATION_BY_ISOTOPE[cleaned]
    for key, val in WATER_LINEAR_ATTENUATION_BY_ISOTOPE.items():
        if key.lower() == cleaned.lower():
            return val
    raise ValueError(
        f"Неизвестный радионуклид '{target_isotope}'. "
        f"Доступные варианты: {list(WATER_LINEAR_ATTENUATION_BY_ISOTOPE.keys())}"
    )


def create_uniform_attenuation_map(
    shape: tuple[int, ...],
    linear_attenuation: float = 0.188 / cm,
    voxel_size: float = 4.2 * mm,
    phantom_radius: float | None = None,
    center_offset: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """
    Формирует однородную 2D или 3D карту линейного коэффициента ослабления
    цилиндрического/кругового фантома в системе единиц `hepunits`.

    :param shape: Форма реконструируемого объёма `(nx, nx)` или `(nx, nx, ny)`.
    :param linear_attenuation: Линейный коэффициент ослабления среды в `hepunits`.
    :param voxel_size: Физический размер вокселя в `hepunits`.
    :param phantom_radius: Физический радиус фантома в `hepunits` (по умолчанию 0.8 от радиуса FOV).
    :param center_offset: Смещение центра фантома `(x, y)` в `hepunits` относительно оси вращения.
    :return: Массив линейных коэффициентов ослабления формы `shape`.
    """
    if len(shape) not in (2, 3):
        raise ValueError(f"Форма карты ослабления должна быть 2D или 3D, получено: {shape}")

    nx, ny_trans = int(shape[0]), int(shape[1])
    if nx != ny_trans:
        raise ValueError(
            f"Трансаксиальные измерения должны быть квадратными (nx == ny), получено: {shape}"
        )

    step = float(voxel_size)
    if step <= 0.0:
        raise ValueError("Размер вокселя voxel_size должен быть строго положительным.")

    center = 0.5 * (nx - 1)
    fov_radius = center * step
    target_radius = float(phantom_radius) if phantom_radius is not None else 0.8 * fov_radius

    offset_x = float(center_offset[0]) if center_offset is not None else 0.0
    offset_y = float(center_offset[1]) if center_offset is not None else 0.0

    coords = (np.arange(nx, dtype=np.float64) - center) * step
    phys_x, phys_y = np.meshgrid(coords - offset_x, coords - offset_y, indexing="ij")
    mask_2d = (phys_x * phys_x + phys_y * phys_y) <= (target_radius * target_radius + 1e-9)

    att_2d = np.where(mask_2d, float(linear_attenuation), 0.0).astype(np.float64)
    if len(shape) == 2:
        return att_2d

    nz = int(shape[2])
    return np.repeat(att_2d[:, :, np.newaxis], nz, axis=2)


def estimate_attenuation_map_from_projections(
    projections: NDArray[np.float64],
    angles: NDArray[np.float64] | Sequence[float] | None = None,
    linear_attenuation: float = 0.188 / cm,
    voxel_size: float = 4.2 * mm,
    threshold_fraction: float = 0.05,
) -> NDArray[np.float64]:
    """
    Автоматически оценивает однородную маску контура тела/фантома по проекционным
    данным (пересечение проекционных теней / Visual Hull) и заполняет её заданным
    линейным коэффициентом ослабления `linear_attenuation`.

    :param projections: Проекционные данные формы `(views, nx)` или `(views, nx, ny)`.
    :param angles: Углы поворота гантри в `hepunits`.
    :param linear_attenuation: Линейный коэффициент ослабления среды в `hepunits`.
    :param voxel_size: Размер вокселя в `hepunits`.
    :param threshold_fraction: Относительный порог отсечения фона проекций `(0..1)`.
    :return: Оценённая карта коэффициентов ослабления формы `(nx, nx)` или `(nx, nx, ny)`.
    """
    proj_arr = np.asarray(projections, dtype=np.float64)
    if proj_arr.ndim not in (2, 3):
        raise ValueError(
            f"Ожидался 2D или 3D массив проекций, получена форма: {proj_arr.shape}"
        )

    num_views = int(proj_arr.shape[0])
    nx = int(proj_arr.shape[1])
    norm_angles = normalize_projection_angles(angles, num_views)

    is_2d = proj_arr.ndim == 2
    proj_3d = proj_arr[:, :, np.newaxis] if is_2d else proj_arr
    ny = int(proj_3d.shape[2])

    max_per_view = np.max(proj_3d, axis=1, keepdims=True)
    global_max = float(np.max(proj_3d))
    if global_max <= 0.0:
        out_shape = (nx, nx) if is_2d else (nx, nx, ny)
        return np.zeros(out_shape, dtype=np.float64)

    thresh = np.maximum(max_per_view * float(threshold_fraction), global_max * 0.01)
    active_mask = proj_3d >= thresh

    # Векторизованно расширяем проекционную поддержку до непрерывного интервала между левым и правым краем
    has_left = np.cumsum(active_mask, axis=1) > 0
    has_right = np.cumsum(active_mask[:, ::-1, :], axis=1)[:, ::-1, :] > 0
    support = (has_left & has_right).astype(np.float64)

    backprojected_support = TomographicProjector.backproject(
        projections=support,
        angles=norm_angles,
        attenuation_map=None,
        voxel_size=voxel_size,
        apply_fov_mask=True,
    )
    # Точка принадлежит выпуклой оболочке объекта, если она попадает в тень на большинстве углов
    hull_mask = backprojected_support >= (0.85 * num_views)
    if not np.any(hull_mask):
        # Fallback на стандартный цилиндр при слишком разреженных данных
        out_shape = (nx, nx) if is_2d else (nx, nx, ny)
        return create_uniform_attenuation_map(
            shape=out_shape,
            linear_attenuation=linear_attenuation,
            voxel_size=voxel_size,
        )

    att_map = np.where(hull_mask, float(linear_attenuation), 0.0).astype(np.float64)
    return att_map[:, :, 0] if is_2d else att_map


def compute_chang_correction_map(
    attenuation_map: NDArray[np.float64],
    angles: NDArray[np.float64] | Sequence[float],
    voxel_size: float = 4.2 * mm,
    apply_fov_mask: bool = True,
) -> NDArray[np.float64]:
    """
    Вычисляет поправочную матрицу ослабления по методу Чанга (Chang's correction map):
    среднее по всем углам проецирования значение коэффициента прохождения гамма-квантов
    `C(r) = (1 / N_views) * sum_k exp(-integral_{L(r, theta_k)} mu(s) ds)`.

    :param attenuation_map: 2D `(nx, nx)` или 3D `(nx, nx, ny)` карта линейного ослабления в `hepunits`.
    :param angles: Массив углов проекций в `hepunits`.
    :param voxel_size: Физический размер вокселя в `hepunits`.
    :param apply_fov_mask: Применять ли круговую маску поля зрения.
    :return: Поправочная карта `C(r)` со значениями в диапазоне `[0, 1]` той же формы, что и `attenuation_map`.
    """
    att_arr = np.asarray(attenuation_map, dtype=np.float64)
    if att_arr.ndim not in (2, 3):
        raise ValueError(
            f"Ожидалась 2D или 3D карта ослабления, получена форма: {att_arr.shape}"
        )

    is_2d = att_arr.ndim == 2
    nx = int(att_arr.shape[0])
    ny = 1 if is_2d else int(att_arr.shape[2])

    angle_seq = np.asarray(angles, dtype=np.float64).ravel()
    num_views = int(angle_seq.size)
    if num_views == 0:
        raise ValueError("Массив углов не может быть пустым.")
    norm_angles = normalize_projection_angles(angle_seq, num_views)

    # Если 3D карта однородна вдоль оси Z, достаточно вычислить 2D карту и размножить её вдоль оси Z
    if not is_2d and ny > 1 and bool(np.all(att_arr[:, :, :1] == att_arr)):
        corr_2d = compute_chang_correction_map(
            attenuation_map=att_arr[:, :, 0],
            angles=norm_angles,
            voxel_size=voxel_size,
            apply_fov_mask=apply_fov_mask,
        )
        return np.repeat(corr_2d[:, :, np.newaxis], ny, axis=2)

    unit_projections = np.ones(
        (num_views, nx) if is_2d else (num_views, nx, ny), dtype=np.float64
    )
    accumulated = TomographicProjector.backproject(
        projections=unit_projections,
        angles=norm_angles,
        attenuation_map=att_arr,
        voxel_size=voxel_size,
        apply_fov_mask=apply_fov_mask,
    )
    correction_map = accumulated / float(num_views)
    return np.clip(correction_map, 0.0, 1.0)


def apply_chang_attenuation_correction(
    reconstructed_volume: NDArray[np.float64],
    attenuation_map: NDArray[np.float64],
    angles: NDArray[np.float64] | Sequence[float],
    voxel_size: float = 4.2 * mm,
    min_correction_factor: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Применяет коррекцию затухания первого порядка по методу Чанга к реконструированному
    2D срезу или 3D объёму путём деления на карту средних коэффициентов прохождения.

    :param reconstructed_volume: Реконструированный объём без коррекции затухания.
    :param attenuation_map: Карта линейных коэффициентов ослабления в `hepunits`.
    :param angles: Вектор углов проекций в `hepunits`.
    :param voxel_size: Размер вокселя в `hepunits`.
    :param min_correction_factor: Минимальный порог коэффициента поправки для защиты от деления на ноль.
    :return: Кортеж `(скорректированный объём, карта коэффициентов коррекции Чанга)`.
    """
    vol_arr = np.asarray(reconstructed_volume, dtype=np.float64)
    att_arr = np.asarray(attenuation_map, dtype=np.float64)
    if att_arr.ndim == 2 and vol_arr.ndim == 3:
        if att_arr.shape != vol_arr.shape[:2]:
            raise ValueError(
                f"Трансаксиальная форма 2D карты затухания {att_arr.shape} "
                f"не совпадает с объёмом {vol_arr.shape}."
            )
        # 2D карта затухания транслируется на все аксиальные срезы 3D объёма
        correction_2d = compute_chang_correction_map(
            attenuation_map=att_arr,
            angles=angles,
            voxel_size=voxel_size,
            apply_fov_mask=True,
        )
        correction_map = np.repeat(correction_2d[:, :, np.newaxis], vol_arr.shape[2], axis=2)
    else:
        if att_arr.shape != vol_arr.shape:
            raise ValueError(
                f"Форма карты затухания {att_arr.shape} не совпадает "
                f"с формой реконструированного объёма {vol_arr.shape}."
            )
        correction_map = compute_chang_correction_map(
            attenuation_map=att_arr,
            angles=angles,
            voxel_size=voxel_size,
            apply_fov_mask=True,
        )

    eff_min_factor = max(float(min_correction_factor), 1e-12)
    safe_divisor = np.maximum(correction_map, eff_min_factor)
    corrected = np.where(correction_map > eff_min_factor, vol_arr / safe_divisor, 0.0)
    return corrected, correction_map
