"""
Статистические итерационные алгоритмы томографической реконструкции ОФЭКТ (OSEM и MLEM).

Реализует:
- контейнер результатов реконструкции `ReconstructionResult`;
- генератор максимально разнесённых по углу упорядоченных подмножеств `generate_ordered_subsets`;
- итерационные реконструкторы `OSEMReconstructor` и `MLEMReconstructor` с поддержкой
  лучевой коррекции затухания (Attenuation Correction), моделирования пространственного
  разрешения (PSF), априорной регуляризации (квадратичной и Total Variation по схеме OSL)
  и гауссовой постобработки.

Все физические величины передаются и обрабатываются строго в системе единиц `hepunits`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hepunits import mm
from numpy.typing import NDArray

from nmspect.reconstruction.attenuation import (
    create_uniform_attenuation_map,
    estimate_attenuation_map_from_projections,
)
from nmspect.reconstruction.projector import (
    TomographicProjector,
    _BackprojectionLookup,
    _RotationLookup,
    create_circular_fov_mask,
    gaussian_smooth_volume,
    normalize_projection_angles,
)


@dataclass
class ReconstructionResult:
    """
    Контейнер результатов томографической реконструкции ОФЭКТ.
    """

    volume: NDArray[np.float64]
    voxel_size: float
    method: str
    attenuation_corrected: bool
    correction_map: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_ordered_subsets(
    num_angles: int,
    num_subsets: int,
) -> list[NDArray[np.int64]]:
    """
    Разбивает индексы проекций `[0 .. num_angles - 1]` на сбалансированные
    упорядоченные подмножества с максимальным угловым разнесением.

    :param num_angles: Общее число угловых ракурсов.
    :param num_subsets: Желаемое количество упорядоченных подмножеств.
    :return: Список массивов индексов ракурсов (`int64`) для каждого подмножества.
    """
    if int(num_angles) < 1:
        raise ValueError(
            f"Общее число угловых ракурсов должно быть не меньше 1, получено: {num_angles}."
        )
    if int(num_subsets) < 1:
        raise ValueError(
            f"Число подмножеств num_subsets должно быть не меньше 1, получено: {num_subsets}."
        )

    effective_subsets = min(int(num_subsets), int(num_angles))
    all_indices = np.arange(int(num_angles), dtype=np.int64)
    raw_subsets = [
        all_indices[subset_offset::effective_subsets]
        for subset_offset in range(effective_subsets)
    ]
    non_empty = [sub for sub in raw_subsets if sub.size > 0]

    # Упорядочиваем подмножества так, чтобы соседние по очереди подмножества
    # отстояли друг от друга как можно дальше по углу
    num_non_empty = len(non_empty)
    if num_non_empty <= 2:
        return non_empty

    ordered_indices: list[int] = []
    step = max(1, num_non_empty // 2)
    visited = [False] * num_non_empty
    curr = 0
    for _ in range(num_non_empty):
        while visited[curr]:
            curr = (curr + 1) % num_non_empty
        ordered_indices.append(curr)
        visited[curr] = True
        curr = (curr + step) % num_non_empty

    return [non_empty[idx] for idx in ordered_indices]


class OSEMReconstructor:
    """
    Статистический итерационный реконструктор методом упорядоченных подмножеств
    максимизации правдоподобия (Ordered Subsets Expectation Maximization, OSEM).

    При `num_subsets=1` математически эквивалентен классическому методу MLEM.
    Поддерживает точную лучевую коррекцию затухания (Attenuation Correction)
    внутри прямого и обратного проекторов, а также опциональное моделирование PSF
    и регуляризацию (квадратичную или TV).
    """

    SUPPORTED_PRIORS: tuple[str, ...] = ("none", "quadratic", "tv")

    def __init__(
        self,
        num_iterations: int = 10,
        num_subsets: int = 4,
        voxel_size: float = 4.2 * mm,
        attenuation_map: NDArray[np.float64] | None = None,
        linear_attenuation: float | None = None,
        phantom_radius: float | None = None,
        post_smoothing_fwhm: float | None = None,
        regularization_epsilon: float = 1e-8,
        apply_fov_mask: bool = True,
        psf_fwhm: float | None = None,
        prior_type: str | None = None,
        prior_beta: float = 0.0,
    ) -> None:
        """
        Инициализирует итерационный реконструктор OSEM / MLEM.

        :param num_iterations: Число полных итераций по всем подмножествам.
        :param num_subsets: Количество упорядоченных подмножеств углов (при 1 — классический MLEM).
        :param voxel_size: Физический размер вокселя в единицах `hepunits`.
        :param attenuation_map: 2D или 3D карта линейных коэффициентов ослабления в `hepunits`.
        :param linear_attenuation: Однородный линейный коэффициент ослабления среды в `hepunits`.
        :param phantom_radius: Физический радиус цилиндрического фантома в `hepunits`.
        :param post_smoothing_fwhm: FWHM 3D/2D гауссова фильтра постобработки в `hepunits`.
        :param regularization_epsilon: Малая константа для предотвращения деления на ноль.
        :param apply_fov_mask: Ограничивать ли область реконструкции круговым полем зрения.
        :param psf_fwhm: Опциональная ширина гауссианы отклика коллиматора-детектора (PSF) в `hepunits`.
        :param prior_type: Тип априорной регуляризации (`None`, `'quadratic'` или `'tv'`).
        :param prior_beta: Весовой коэффициент регуляризации One-Step-Late (по умолчанию `0.0`).
        """
        if int(num_iterations) < 1:
            raise ValueError("Число итераций num_iterations должно быть не меньше 1.")
        if int(num_subsets) < 1:
            raise ValueError("Число подмножеств num_subsets должно быть не меньше 1.")
        if float(voxel_size) <= 0.0:
            raise ValueError("Размер вокселя voxel_size должен быть строго положительным.")
        if float(regularization_epsilon) <= 0.0:
            raise ValueError(
                "Параметр регуляризации regularization_epsilon должен быть строго положительным."
            )
        if float(prior_beta) < 0.0:
            raise ValueError("Вес регуляризации prior_beta не может быть отрицательным.")

        norm_prior = "none" if prior_type is None else prior_type.strip().lower()
        if norm_prior not in self.SUPPORTED_PRIORS:
            raise ValueError(
                f"Неподдерживаемый тип регуляризации '{prior_type}'. "
                f"Допустимые варианты: {self.SUPPORTED_PRIORS}"
            )

        self.num_iterations: int = int(num_iterations)
        self.num_subsets: int = int(num_subsets)
        self.voxel_size: float = float(voxel_size)
        self.attenuation_map: NDArray[np.float64] | None = attenuation_map
        self.linear_attenuation: float | None = (
            float(linear_attenuation) if linear_attenuation is not None else None
        )
        self.phantom_radius: float | None = (
            float(phantom_radius) if phantom_radius is not None else None
        )
        self.post_smoothing_fwhm: float | None = (
            float(post_smoothing_fwhm) if post_smoothing_fwhm is not None else None
        )
        self.regularization_epsilon: float = float(regularization_epsilon)
        self.apply_fov_mask: bool = bool(apply_fov_mask)
        self.psf_fwhm: float | None = float(psf_fwhm) if psf_fwhm is not None else None
        self.prior_type: str = norm_prior
        self.prior_beta: float = float(prior_beta)

    def build_ordered_subsets(self, num_views: int) -> list[NDArray[np.int64]]:
        """
        Разбивает индексы проекций `[0 .. num_views - 1]` на сбалансированные
        упорядоченные подмножества с максимальным угловым разнесением.

        :param num_views: Общее число угловых ракурсов.
        :return: Список массивов индексов ракурсов для каждого подмножества.
        """
        return generate_ordered_subsets(
            num_angles=num_views,
            num_subsets=self.num_subsets,
        )

    def _compute_prior_gradient_3d(
        self,
        estimate_3d: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Вычисляет градиент априорного потенциала гиббсовской регуляризации
        (квадратичной или Total Variation) для схемы One-Step-Late (OSL).
        """
        if self.prior_type == "none" or self.prior_beta <= 0.0:
            return np.zeros_like(estimate_3d)

        grad = np.zeros_like(estimate_3d)
        active_axes = (0, 1) if estimate_3d.shape[2] == 1 else (0, 1, 2)
        tv_eps = max(self.regularization_epsilon, 1e-6)

        for axis_idx in active_axes:
            shifted_plus = np.roll(estimate_3d, shift=-1, axis=axis_idx)
            shifted_minus = np.roll(estimate_3d, shift=1, axis=axis_idx)
            diff_plus = estimate_3d - shifted_plus
            diff_minus = estimate_3d - shifted_minus
            if self.prior_type == "quadratic":
                grad += diff_plus + diff_minus
            elif self.prior_type == "tv":
                grad += diff_plus / np.sqrt(diff_plus * diff_plus + tv_eps)
                grad += diff_minus / np.sqrt(diff_minus * diff_minus + tv_eps)
        return grad

    def _resolve_attenuation_map(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64],
        override_attenuation_map: NDArray[np.float64] | None,
        active_voxel_size: float,
    ) -> NDArray[np.float64] | None:
        """
        Определяет актуальную карту ослабления для итерационной реконструкции.
        """
        if override_attenuation_map is not None:
            return np.asarray(override_attenuation_map, dtype=np.float64)
        if self.attenuation_map is not None:
            return np.asarray(self.attenuation_map, dtype=np.float64)
        if self.linear_attenuation is not None and self.linear_attenuation > 0.0:
            nx = int(projections.shape[1])
            out_shape: tuple[int, ...] = (
                (nx, nx) if projections.ndim == 2 else (nx, nx, int(projections.shape[2]))
            )
            if self.phantom_radius is not None:
                return create_uniform_attenuation_map(
                    shape=out_shape,
                    linear_attenuation=self.linear_attenuation,
                    voxel_size=active_voxel_size,
                    phantom_radius=self.phantom_radius,
                )
            return estimate_attenuation_map_from_projections(
                projections=projections,
                angles=angles,
                linear_attenuation=self.linear_attenuation,
                voxel_size=active_voxel_size,
            )
        return None

    def reconstruct_with_metadata(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float] | None = None,
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float | None = None,
        initial_estimate: NDArray[np.float64] | None = None,
    ) -> ReconstructionResult:
        """
        Выполняет итерационную реконструкцию OSEM / MLEM с предвычислением таблиц
        поворота, лучевых интегралов ослабления `exp(-integral(mu ds))` и карт
        чувствительности для каждого подмножества.

        :param projections: Стек измеренных проекций `(views, nx)` или `(views, nx, ny)`.
        :param angles: Вектор углов проекций в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Опциональный размер вокселя в `hepunits`.
        :param initial_estimate: Опциональное начальное приближение объёма.
        :return: Объект `ReconstructionResult`.
        """
        proj_arr = np.asarray(projections, dtype=np.float64)
        if proj_arr.ndim not in (2, 3):
            raise ValueError(
                f"Ожидался 2D или 3D массив проекций, получена форма: {proj_arr.shape}"
            )

        is_2d = proj_arr.ndim == 2
        proj_3d = np.clip(proj_arr[:, :, np.newaxis] if is_2d else proj_arr, 0.0, None)
        num_views, nx, ny = proj_3d.shape

        active_voxel_size = float(voxel_size) if voxel_size is not None else self.voxel_size
        norm_angles = normalize_projection_angles(angles, num_views)

        resolved_att = self._resolve_attenuation_map(
            projections=proj_arr,
            angles=norm_angles,
            override_attenuation_map=attenuation_map,
            active_voxel_size=active_voxel_size,
        )
        att_3d = TomographicProjector._prepare_attenuation_3d(resolved_att, nx, ny)

        fov_mask = (
            create_circular_fov_mask(nx) if self.apply_fov_mask else np.ones((nx, nx), dtype=bool)
        )
        fov_mask_3d = fov_mask[:, :, np.newaxis]

        # Предвычисление таблиц интерполяции и весов ослабления для всех ракурсов
        forward_lookups: list[_RotationLookup] = []
        bp_lookups: list[_BackprojectionLookup] = []
        ray_att_list: list[NDArray[np.float64] | None] = []
        voxel_att_list: list[NDArray[np.float64] | None] = []

        for view_idx in range(num_views):
            angle_val = float(norm_angles[view_idx])
            fwd_lkp = TomographicProjector._build_forward_rotation_lookup(nx, angle_val)
            bp_lkp = TomographicProjector._build_backprojection_1d_lookup(nx, angle_val)
            forward_lookups.append(fwd_lkp)
            bp_lookups.append(bp_lkp)

            if att_3d is not None:
                inv_lkp = TomographicProjector._build_inverse_rotation_lookup(nx, angle_val)
                ray_att = TomographicProjector.compute_ray_attenuation_factors(
                    attenuation_3d=att_3d,
                    forward_lookup=fwd_lkp,
                    voxel_size=active_voxel_size,
                )
                vox_att = TomographicProjector._apply_rotation_lookup_3d(ray_att, inv_lkp)
                vox_att = np.clip(vox_att, 0.0, 1.0)
                ray_att_list.append(ray_att)
                voxel_att_list.append(vox_att)
            else:
                ray_att_list.append(None)
                voxel_att_list.append(None)

        subsets = self.build_ordered_subsets(num_views)

        # Предвычисление карт чувствительности для каждого подмножества
        sensitivity_maps: list[NDArray[np.float64]] = []
        total_sensitivity = np.zeros((nx, nx, ny), dtype=np.float64)
        for subset in subsets:
            sub_sens = np.zeros((nx, nx, ny), dtype=np.float64)
            for view_idx in subset:
                bp_lkp = bp_lookups[int(view_idx)]
                unit_bp = (bp_lkp.w0 + bp_lkp.w1)[:, :, np.newaxis]
                vox_att_item = voxel_att_list[int(view_idx)]
                if vox_att_item is not None:
                    unit_bp = unit_bp * vox_att_item
                sub_sens += unit_bp
            sub_sens *= fov_mask_3d
            sensitivity_maps.append(sub_sens)
            total_sensitivity += sub_sens

        # Начальное приближение: однородное положительное распределение внутри FOV,
        # масштабированное по среднему числу отсчётов на проекцию
        expected_shape = (nx, nx) if is_2d else (nx, nx, ny)
        if initial_estimate is not None:
            est_arr = np.asarray(initial_estimate, dtype=np.float64)
            if est_arr.shape != expected_shape:
                raise ValueError(
                    f"Неверная форма начального приближения initial_estimate {est_arr.shape}, "
                    f"ожидалось {expected_shape}."
                )
            estimate = (
                est_arr[:, :, np.newaxis] if est_arr.ndim == 2 else est_arr.copy()
            ) * fov_mask_3d
            estimate = np.clip(estimate, self.regularization_epsilon, None) * fov_mask_3d
        else:
            active_voxels_per_slice = max(float(np.sum(fov_mask)), 1.0)
            mean_counts_per_view = np.mean(np.sum(proj_3d, axis=1), axis=0)
            init_slice_vals = np.maximum(mean_counts_per_view / active_voxels_per_slice, 1e-4)
            estimate = np.broadcast_to(
                init_slice_vals[np.newaxis, np.newaxis, :], (nx, nx, ny)
            ).copy()
            estimate *= fov_mask_3d

        eps = self.regularization_epsilon
        out_sens = total_sensitivity[:, :, 0] if is_2d else total_sensitivity
        method_label = "MLEM" if len(subsets) == 1 else "OSEM"
        active_psf_fwhm: float | None = (
            self.psf_fwhm if (self.psf_fwhm is not None and self.psf_fwhm > 0.0) else None
        )

        # Основной итерационный цикл OSEM / MLEM
        for _ in range(self.num_iterations):
            for sub_idx, subset in enumerate(subsets):
                sub_sens = sensitivity_maps[sub_idx]
                backprojected_ratio = np.zeros((nx, nx, ny), dtype=np.float64)

                source_est = (
                    gaussian_smooth_volume(
                        volume=estimate[:, :, 0] if is_2d else estimate,
                        fwhm=active_psf_fwhm,
                        voxel_size=active_voxel_size,
                    )
                    if active_psf_fwhm is not None
                    else estimate
                )
                if active_psf_fwhm is not None and is_2d:
                    source_est = source_est[:, :, np.newaxis]

                for view_idx in subset:
                    v_idx = int(view_idx)
                    fwd_lkp = forward_lookups[v_idx]
                    bp_lkp = bp_lookups[v_idx]

                    # 1. Прямое проецирование текущей оценки с учётом затухания
                    rot_est = TomographicProjector._apply_rotation_lookup_3d(source_est, fwd_lkp)
                    ray_att_item = ray_att_list[v_idx]
                    if ray_att_item is not None:
                        rot_est = rot_est * ray_att_item
                    expected_proj = np.sum(rot_est, axis=1)

                    # 2. Отношение измеренной проекции к ожидаемой P / (H * f + eps)
                    measured_proj = proj_3d[v_idx]
                    ratio = np.where(
                        expected_proj > eps,
                        measured_proj / (expected_proj + eps),
                        0.0,
                    )

                    # 3. Обратное проецирование отношения с весом затухания H^T * ratio
                    bp_view = (
                        bp_lkp.w0[:, :, np.newaxis] * ratio[bp_lkp.u0, :]
                        + bp_lkp.w1[:, :, np.newaxis] * ratio[bp_lkp.u1, :]
                    )
                    vox_att_view = voxel_att_list[v_idx]
                    if vox_att_view is not None:
                        bp_view = bp_view * vox_att_view
                    backprojected_ratio += bp_view

                if active_psf_fwhm is not None:
                    smoothed_bp = gaussian_smooth_volume(
                        volume=backprojected_ratio[:, :, 0] if is_2d else backprojected_ratio,
                        fwhm=active_psf_fwhm,
                        voxel_size=active_voxel_size,
                    )
                    backprojected_ratio = (
                        smoothed_bp[:, :, np.newaxis] if is_2d else smoothed_bp
                    )

                # 4. Мультипликативное обновление оценки в области положительной чувствительности
                eff_denom = sub_sens
                if self.prior_type != "none" and self.prior_beta > 0.0:
                    prior_grad = self._compute_prior_gradient_3d(estimate)
                    eff_denom = np.maximum(sub_sens + self.prior_beta * prior_grad, eps)
                valid_sens = sub_sens > eps
                update_factor = np.where(
                    valid_sens, backprojected_ratio / np.maximum(eff_denom, eps), 0.0
                )
                estimate = np.asarray(
                    np.where(valid_sens, estimate * update_factor, 0.0) * fov_mask_3d,
                    dtype=np.float64,
                )

        if self.post_smoothing_fwhm is not None and self.post_smoothing_fwhm > 0.0:
            estimate = gaussian_smooth_volume(
                volume=estimate,
                fwhm=self.post_smoothing_fwhm,
                voxel_size=active_voxel_size,
            )
            estimate *= fov_mask_3d

        out_vol = estimate[:, :, 0] if is_2d else estimate

        return ReconstructionResult(
            volume=out_vol,
            voxel_size=active_voxel_size,
            method=method_label,
            attenuation_corrected=resolved_att is not None,
            correction_map=out_sens,
            metadata={
                "num_iterations": self.num_iterations,
                "num_subsets": len(subsets),
                "post_smoothing_fwhm": self.post_smoothing_fwhm,
            },
        )

    def reconstruct_iterations(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float] | None = None,
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float | None = None,
        initial_estimate: NDArray[np.float64] | None = None,
        iterations_to_save: Sequence[int] | None = None,
    ) -> dict[int, ReconstructionResult]:
        """
        Выполняет итерационную реконструкцию OSEM / MLEM с единовременным предвычислением
        геометрических таблиц и весов затухания, возвращая словарь результатов для каждой
        запрошенной итерации (по умолчанию `1 .. self.num_iterations`).

        :param projections: Стек измеренных проекций `(views, nx)` или `(views, nx, ny)`.
        :param angles: Вектор углов проекций в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Опциональный размер вокселя в `hepunits`.
        :param initial_estimate: Опциональное начальное приближение объёма.
        :param iterations_to_save: Список номеров итераций `(1..N)` для сохранения.
        :return: Словарь `{номер_итерации: ReconstructionResult}`.
        """
        proj_arr = np.asarray(projections, dtype=np.float64)
        if proj_arr.ndim not in (2, 3):
            raise ValueError(
                f"Ожидался 2D или 3D массив проекций, получена форма: {proj_arr.shape}"
            )

        target_iters = (
            sorted({int(iter_num) for iter_num in iterations_to_save if int(iter_num) >= 1})
            if iterations_to_save is not None
            else list(range(1, self.num_iterations + 1))
        )
        max_iter = max(target_iters) if target_iters else self.num_iterations
        target_set = set(target_iters)

        is_2d = proj_arr.ndim == 2
        proj_3d = np.clip(proj_arr[:, :, np.newaxis] if is_2d else proj_arr, 0.0, None)
        num_views, nx, ny = proj_3d.shape

        active_voxel_size = float(voxel_size) if voxel_size is not None else self.voxel_size
        norm_angles = normalize_projection_angles(angles, num_views)

        resolved_att = self._resolve_attenuation_map(
            projections=proj_arr,
            angles=norm_angles,
            override_attenuation_map=attenuation_map,
            active_voxel_size=active_voxel_size,
        )
        att_3d = TomographicProjector._prepare_attenuation_3d(resolved_att, nx, ny)

        fov_mask = (
            create_circular_fov_mask(nx) if self.apply_fov_mask else np.ones((nx, nx), dtype=bool)
        )
        fov_mask_3d = fov_mask[:, :, np.newaxis]

        forward_lookups: list[_RotationLookup] = []
        bp_lookups: list[_BackprojectionLookup] = []
        ray_att_list: list[NDArray[np.float64] | None] = []
        voxel_att_list: list[NDArray[np.float64] | None] = []

        for view_idx in range(num_views):
            angle_val = float(norm_angles[view_idx])
            fwd_lkp = TomographicProjector._build_forward_rotation_lookup(nx, angle_val)
            bp_lkp = TomographicProjector._build_backprojection_1d_lookup(nx, angle_val)
            forward_lookups.append(fwd_lkp)
            bp_lookups.append(bp_lkp)

            if att_3d is not None:
                inv_lkp = TomographicProjector._build_inverse_rotation_lookup(nx, angle_val)
                ray_att = TomographicProjector.compute_ray_attenuation_factors(
                    attenuation_3d=att_3d,
                    forward_lookup=fwd_lkp,
                    voxel_size=active_voxel_size,
                )
                vox_att = TomographicProjector._apply_rotation_lookup_3d(ray_att, inv_lkp)
                vox_att = np.clip(vox_att, 0.0, 1.0)
                ray_att_list.append(ray_att)
                voxel_att_list.append(vox_att)
            else:
                ray_att_list.append(None)
                voxel_att_list.append(None)

        subsets = self.build_ordered_subsets(num_views)
        sensitivity_maps: list[NDArray[np.float64]] = []
        total_sensitivity = np.zeros((nx, nx, ny), dtype=np.float64)
        for subset in subsets:
            sub_sens = np.zeros((nx, nx, ny), dtype=np.float64)
            for view_idx in subset:
                bp_lkp = bp_lookups[int(view_idx)]
                unit_bp = (bp_lkp.w0 + bp_lkp.w1)[:, :, np.newaxis]
                vox_att_item = voxel_att_list[int(view_idx)]
                if vox_att_item is not None:
                    unit_bp = unit_bp * vox_att_item
                sub_sens += unit_bp
            sub_sens *= fov_mask_3d
            sensitivity_maps.append(sub_sens)
            total_sensitivity += sub_sens

        expected_shape = (nx, nx) if is_2d else (nx, nx, ny)
        if initial_estimate is not None:
            est_arr = np.asarray(initial_estimate, dtype=np.float64)
            if est_arr.shape != expected_shape:
                raise ValueError(
                    f"Неверная форма начального приближения initial_estimate {est_arr.shape}, "
                    f"ожидалось {expected_shape}."
                )
            estimate = (
                est_arr[:, :, np.newaxis] if est_arr.ndim == 2 else est_arr.copy()
            ) * fov_mask_3d
            estimate = np.clip(estimate, self.regularization_epsilon, None) * fov_mask_3d
        else:
            active_voxels_per_slice = max(float(np.sum(fov_mask)), 1.0)
            mean_counts_per_view = np.mean(np.sum(proj_3d, axis=1), axis=0)
            init_slice_vals = np.maximum(mean_counts_per_view / active_voxels_per_slice, 1e-4)
            estimate = np.broadcast_to(
                init_slice_vals[np.newaxis, np.newaxis, :], (nx, nx, ny)
            ).copy()
            estimate *= fov_mask_3d

        eps = self.regularization_epsilon
        out_sens = total_sensitivity[:, :, 0] if is_2d else total_sensitivity
        method_label = "MLEM" if len(subsets) == 1 else "OSEM"
        results_by_iter: dict[int, ReconstructionResult] = {}
        active_psf_fwhm: float | None = (
            self.psf_fwhm if (self.psf_fwhm is not None and self.psf_fwhm > 0.0) else None
        )

        for iter_idx in range(1, max_iter + 1):
            for sub_idx, subset in enumerate(subsets):
                sub_sens = sensitivity_maps[sub_idx]
                backprojected_ratio = np.zeros((nx, nx, ny), dtype=np.float64)

                source_est = (
                    gaussian_smooth_volume(
                        volume=estimate[:, :, 0] if is_2d else estimate,
                        fwhm=active_psf_fwhm,
                        voxel_size=active_voxel_size,
                    )
                    if active_psf_fwhm is not None
                    else estimate
                )
                if active_psf_fwhm is not None and is_2d:
                    source_est = source_est[:, :, np.newaxis]

                for view_idx in subset:
                    v_idx = int(view_idx)
                    fwd_lkp = forward_lookups[v_idx]
                    bp_lkp = bp_lookups[v_idx]

                    rot_est = TomographicProjector._apply_rotation_lookup_3d(source_est, fwd_lkp)
                    ray_att_item = ray_att_list[v_idx]
                    if ray_att_item is not None:
                        rot_est = rot_est * ray_att_item
                    expected_proj = np.sum(rot_est, axis=1)

                    measured_proj = proj_3d[v_idx]
                    ratio = np.where(
                        expected_proj > eps,
                        measured_proj / (expected_proj + eps),
                        0.0,
                    )

                    bp_view = (
                        bp_lkp.w0[:, :, np.newaxis] * ratio[bp_lkp.u0, :]
                        + bp_lkp.w1[:, :, np.newaxis] * ratio[bp_lkp.u1, :]
                    )
                    vox_att_view = voxel_att_list[v_idx]
                    if vox_att_view is not None:
                        bp_view = bp_view * vox_att_view
                    backprojected_ratio += bp_view

                if active_psf_fwhm is not None:
                    smoothed_bp = gaussian_smooth_volume(
                        volume=backprojected_ratio[:, :, 0] if is_2d else backprojected_ratio,
                        fwhm=active_psf_fwhm,
                        voxel_size=active_voxel_size,
                    )
                    backprojected_ratio = (
                        smoothed_bp[:, :, np.newaxis] if is_2d else smoothed_bp
                    )

                eff_denom = sub_sens
                if self.prior_type != "none" and self.prior_beta > 0.0:
                    prior_grad = self._compute_prior_gradient_3d(estimate)
                    eff_denom = np.maximum(sub_sens + self.prior_beta * prior_grad, eps)
                valid_sens = sub_sens > eps
                update_factor = np.where(
                    valid_sens, backprojected_ratio / np.maximum(eff_denom, eps), 0.0
                )
                estimate = np.asarray(
                    np.where(valid_sens, estimate * update_factor, 0.0) * fov_mask_3d,
                    dtype=np.float64,
                )

            if iter_idx in target_set:
                snap = estimate.copy()
                if self.post_smoothing_fwhm is not None and self.post_smoothing_fwhm > 0.0:
                    snap = gaussian_smooth_volume(
                        volume=snap,
                        fwhm=self.post_smoothing_fwhm,
                        voxel_size=active_voxel_size,
                    )
                    snap *= fov_mask_3d
                out_vol = snap[:, :, 0] if is_2d else snap
                results_by_iter[iter_idx] = ReconstructionResult(
                    volume=out_vol,
                    voxel_size=active_voxel_size,
                    method=method_label,
                    attenuation_corrected=resolved_att is not None,
                    correction_map=out_sens,
                    metadata={
                        "num_iterations": iter_idx,
                        "num_subsets": len(subsets),
                        "post_smoothing_fwhm": self.post_smoothing_fwhm,
                    },
                )

        return results_by_iter

    def reconstruct(
        self,
        projections: NDArray[np.float64],
        angles: NDArray[np.float64] | Sequence[float] | None = None,
        attenuation_map: NDArray[np.float64] | None = None,
        voxel_size: float | None = None,
        initial_estimate: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Выполняет итерационную реконструкцию OSEM / MLEM и возвращает восстановленный
        2D срез `(nx, nx)` или 3D объём `(nx, nx, ny)`.

        :param projections: Стек проекций `(views, nx)` или `(views, nx, ny)`.
        :param angles: Углы поворота гантри в `hepunits`.
        :param attenuation_map: Опциональная карта линейного ослабления в `hepunits`.
        :param voxel_size: Опциональный размер вокселя в `hepunits`.
        :param initial_estimate: Опциональное начальное приближение.
        :return: Восстановленное распределение активности.
        """
        return self.reconstruct_with_metadata(
            projections=projections,
            angles=angles,
            attenuation_map=attenuation_map,
            voxel_size=voxel_size,
            initial_estimate=initial_estimate,
        ).volume


class MLEMReconstructor(OSEMReconstructor):
    """
    Классический итерационный реконструктор методом максимизации правдоподобия
    (Maximum Likelihood Expectation Maximization, MLEM, частный случай OSEM при `num_subsets=1`).
    """

    def __init__(
        self,
        num_iterations: int = 20,
        voxel_size: float = 4.2 * mm,
        attenuation_map: NDArray[np.float64] | None = None,
        linear_attenuation: float | None = None,
        phantom_radius: float | None = None,
        post_smoothing_fwhm: float | None = None,
        regularization_epsilon: float = 1e-8,
        apply_fov_mask: bool = True,
        psf_fwhm: float | None = None,
        prior_type: str | None = None,
        prior_beta: float = 0.0,
    ) -> None:
        super().__init__(
            num_iterations=num_iterations,
            num_subsets=1,
            voxel_size=voxel_size,
            attenuation_map=attenuation_map,
            linear_attenuation=linear_attenuation,
            phantom_radius=phantom_radius,
            post_smoothing_fwhm=post_smoothing_fwhm,
            regularization_epsilon=regularization_epsilon,
            apply_fov_mask=apply_fov_mask,
            psf_fwhm=psf_fwhm,
            prior_type=prior_type,
            prior_beta=prior_beta,
        )
