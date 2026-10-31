"""Расчёт весовых коэффициентов вычитания комптоновского рассеяния (методы DEW и TEW)."""

from __future__ import annotations


def calculate_tew_coefficients(
    main_lower: float,
    main_upper: float,
    low_scatter_lower: float,
    low_scatter_upper: float,
    high_scatter_lower: float,
    high_scatter_upper: float,
) -> tuple[float, float]:
    """Вычисляет весовые коэффициенты для метода тройного энергетического окна (TEW, Ogawa et al., 1991).

    Оценка счёта рассеянных фотонов в фотопике по трапецеидальной аппроксимации:
        C_scatter = (C_low / W_low + C_high / W_high) * (W_main / 2)
    Поэтому эффективные коэффициенты при суммировании C_primary = C_main + w_low * C_low + w_high * C_high
    равны:
        w_low = -0.5 * (W_main / W_low)
        w_high = -0.5 * (W_main / W_high)

    Args:
        main_lower: Нижняя граница основного окна фотопика.
        main_upper: Верхняя граница основного окна фотопика.
        low_scatter_lower: Нижняя граница нижнего окна рассеяния.
        low_scatter_upper: Верхняя граница нижнего окна рассеяния.
        high_scatter_lower: Нижняя граница верхнего окна рассеяния.
        high_scatter_upper: Верхняя граница верхнего окна рассеяния.

    Returns:
        Кортеж отрицательных весовых коэффициентов (weight_low, weight_high).
    """
    width_main = main_upper - main_lower
    width_low = low_scatter_upper - low_scatter_lower
    width_high = high_scatter_upper - high_scatter_lower

    if width_main <= 0.0 or width_low <= 0.0 or width_high <= 0.0:
        raise ValueError("Ширина каждого энергетического окна в методе TEW должна быть строго положительной.")

    weight_low = -0.5 * (width_main / width_low)
    weight_high = -0.5 * (width_main / width_high)
    return weight_low, weight_high


def calculate_dew_coefficient(
    main_lower: float,
    main_upper: float,
    scatter_lower: float,
    scatter_upper: float,
    k_factor: float = 1.0,
) -> float:
    """Вычисляет весовой коэффициент для метода двойного энергетического окна (DEW, Jaszczak et al., 1984).

    Формула коррекции:
        C_primary = C_main - k_factor * (W_main / W_scatter) * C_scatter

    Args:
        main_lower: Нижняя граница основного окна фотопика.
        main_upper: Верхняя граница основного окна фотопика.
        scatter_lower: Нижняя граница окна рассеяния.
        scatter_upper: Верхняя граница окна рассеяния.
        k_factor: Эмпирический коэффициент масштабирования рассеяния (по умолчанию 1.0 для 50% трапеции с нулём справа: 0.5).

    Returns:
        Отрицательный весовой коэффициент вычитания для окна рассеяния.
    """
    width_main = main_upper - main_lower
    width_scatter = scatter_upper - scatter_lower

    if width_main <= 0.0 or width_scatter <= 0.0:
        raise ValueError("Ширина основного окна и окна рассеяния в методе DEW должна быть строго положительной.")
    if k_factor <= 0.0:
        raise ValueError(f"Коэффициент k_factor должен быть положительным, получено: {k_factor}")

    return -k_factor * (width_main / width_scatter)
