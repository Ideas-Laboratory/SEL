import numpy as np
import sys
import torch
import time
from .ibFFT_CPU_NumbaKernel import *
from pyfftw.interfaces.numpy_fft import rfft2, irfft2
import pyfftw
from numba import njit, types
from numba.typed import Dict
# import numba
pyfftw.interfaces.cache.enable()


# pyfftw.config.NUM_THREADS = 8
# numba.set_num_threads(8)


def ibFFT_repulsive(Y, n_interpolation_points, intervals_per_integer, min_num_intervals, gamma, paraFactor):
    max_coord = Y.max()
    min_coord = Y.min()
    N = Y.shape[0]
    n_boxes_per_dim = int(np.minimum(np.sqrt(16 * N), np.maximum(np.sqrt(4 * N / np.log(N)), np.maximum(
        min_num_intervals, (max_coord.item() - min_coord.item()) / intervals_per_integer))))
    allowed_n_boxes_per_dim = np.array([50, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243,
                                        256, 288, 324, 384, 432, 576, 648, 768, 864, 972, 1152, 1296])
    if (n_boxes_per_dim < allowed_n_boxes_per_dim[-1]):
        n_boxes_per_dim = allowed_n_boxes_per_dim[(
                allowed_n_boxes_per_dim > n_boxes_per_dim)][0]
    else:
        n_boxes_per_dim = allowed_n_boxes_per_dim[-1]
    n_boxes_per_dim = n_boxes_per_dim.item()
    squared_n_terms = 3
    n_terms = squared_n_terms
    ChargesQij = np.ones((N, squared_n_terms), dtype=np.float32)
    ChargesQij[:, :2] = Y

    box_width = (max_coord - min_coord) / n_boxes_per_dim
    n_boxes = n_boxes_per_dim * n_boxes_per_dim
    n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim
    n_fft_coeffs = 2 * n_interpolation_points_1d

    whsquare = box_width / n_interpolation_points
    whsquare *= whsquare
    h = 1.0 / n_interpolation_points * box_width
    y_tilde_spacings = np.array(
        np.arange(n_interpolation_points) * h + h / 2, dtype=np.float32)

    if int(gamma) == gamma:
        half_kernel = paraFactor / ((1.0 + (np.arange(n_interpolation_points_1d) ** 2 + (
                np.arange(n_interpolation_points_1d) ** 2).reshape(-1, 1)) * whsquare) ** int(gamma))
    else:
        half_kernel = paraFactor * np.power(1.0 + (np.arange(n_interpolation_points_1d) ** 2 + (
                np.arange(n_interpolation_points_1d) ** 2).reshape(-1, 1)) * whsquare, -gamma)
    circulant_kernel_tilde = np.zeros(
        (n_fft_coeffs, n_fft_coeffs), dtype=np.float32)
    circulant_kernel_tilde[n_interpolation_points_1d:,
    n_interpolation_points_1d:] = half_kernel
    circulant_kernel_tilde[1:n_interpolation_points_1d + 1,
    n_interpolation_points_1d:] = np.flipud(half_kernel)
    circulant_kernel_tilde[n_interpolation_points_1d:,
    1:n_interpolation_points_1d + 1] = np.fliplr(half_kernel)
    circulant_kernel_tilde[1:n_interpolation_points_1d + 1,
    1:n_interpolation_points_1d + 1] = np.fliplr(np.flipud(half_kernel))
    fft_kernel_tilde = rfft2(circulant_kernel_tilde)
    # fft_kernel_tilde = rfft2(circulant_kernel_tilde,threads=4)
    box_idx = np.ndarray((N, 2), dtype=np.int32)

    Box_idx(box_idx, Y, box_width, min_coord, n_boxes_per_dim, N)

    y_in_box = np.zeros_like(Y)

    Y_in_box(y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N)

    denominator_sub = (y_tilde_spacings.reshape(-1, 1) - y_tilde_spacings)
    np.fill_diagonal(denominator_sub, 1)
    denominator = denominator_sub.prod(axis=0)

    interpolate_values = np.ndarray(
        (N, n_interpolation_points, 2), dtype=np.float32)
    Interpolate(y_in_box, y_tilde_spacings, denominator,
                interpolate_values, n_interpolation_points, N)

    w_coefficients = np.zeros((n_boxes_per_dim * n_interpolation_points,
                               n_boxes_per_dim * n_interpolation_points, squared_n_terms), dtype=np.float32)

    Compute_w_coeff(w_coefficients, box_idx, ChargesQij, interpolate_values,
                    n_interpolation_points, n_boxes_per_dim, n_terms, N)

    mat_w = np.zeros((2 * n_boxes_per_dim * n_interpolation_points, 2 *
                      n_boxes_per_dim * n_interpolation_points, n_terms), dtype=np.float32)
    mat_w[:n_boxes_per_dim * n_interpolation_points,
    :n_boxes_per_dim * n_interpolation_points] = w_coefficients
    mat_w = mat_w.transpose((2, 0, 1))
    # fft_w = rfft2(mat_w,threads=4)
    fft_w = rfft2(mat_w)
    rmut = fft_w * fft_kernel_tilde
    # output = irfft2(rmut,threads=4)
    output = irfft2(rmut)
    potentialsQij = np.zeros((N, n_terms), dtype=np.float32)
    PotentialsQij(potentialsQij, box_idx, interpolate_values,
                  output, n_interpolation_points, n_boxes_per_dim, n_terms, N)
    neg_f = np.ndarray((N, 2), dtype=np.float32)
    PotentialsCom = potentialsQij[:, 2].reshape((-1, 1))
    PotentialsXY = potentialsQij[:, :2]
    neg_f = PotentialsCom * Y - PotentialsXY
    return neg_f


def ibFFT_repulsive_super(Y, n_interpolation_points, intervals_per_integer, min_num_intervals, gamma, paraFactor,
                          superid):
    max_coord = Y.max()
    min_coord = Y.min()
    N = Y.shape[0]
    n_boxes_per_dim = int(np.minimum(np.sqrt(16 * N), np.maximum(np.sqrt(4 * N / np.log(N)), np.maximum(
        min_num_intervals, (max_coord.item() - min_coord.item()) / intervals_per_integer))))
    allowed_n_boxes_per_dim = np.array([50, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243,
                                        256, 288, 324, 384, 432, 576, 648, 768, 864, 972, 1152, 1296])
    if n_boxes_per_dim < allowed_n_boxes_per_dim[-1]:
        n_boxes_per_dim = allowed_n_boxes_per_dim[(
                allowed_n_boxes_per_dim > n_boxes_per_dim)][0]
    else:
        n_boxes_per_dim = allowed_n_boxes_per_dim[-1]
    n_boxes_per_dim = n_boxes_per_dim.item()
    squared_n_terms = 3
    n_terms = squared_n_terms
    ChargesQij = np.ones((N, squared_n_terms), dtype=np.float32)
    ChargesQij[:, :2] = Y

    box_width = (max_coord - min_coord) / n_boxes_per_dim
    n_boxes = n_boxes_per_dim * n_boxes_per_dim
    n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim
    n_fft_coeffs = 2 * n_interpolation_points_1d

    whsquare = box_width / n_interpolation_points
    whsquare *= whsquare
    h = 1.0 / n_interpolation_points * box_width
    y_tilde_spacings = np.array(
        np.arange(n_interpolation_points) * h + h / 2, dtype=np.float32)

    # 创建半核，针对 superid 调整 gamma
    if int(gamma) == gamma:
        # 初始化默认的 gamma
        base_gamma = gamma
        gamma_matrix = np.ones((n_interpolation_points_1d, n_interpolation_points_1d), dtype=np.float32) * base_gamma

        # 针对 superid 中的点，调整斥力系数
        for idx in range(N):
            if idx in superid:
                gamma_matrix[idx, :] *= 1.2  # 将 gamma 放大 n 倍
                gamma_matrix[:, idx] *= 1.2

        half_kernel = paraFactor / ((1.0 + (np.arange(n_interpolation_points_1d) ** 2 +
                                            (np.arange(n_interpolation_points_1d) ** 2).reshape(-1,
                                                                                                1)) * whsquare) ** gamma_matrix)
    else:
        base_gamma = gamma
        gamma_matrix = np.ones((n_interpolation_points_1d, n_interpolation_points_1d), dtype=np.float32) * base_gamma

        # 针对 superid 中的点，调整斥力系数
        for idx in range(N):
            if idx in superid:
                gamma_matrix[idx, :] *= 1.2  # 将 gamma 放大 100 倍
                gamma_matrix[:, idx] *= 1.2

        half_kernel = paraFactor * np.power(1.0 + (np.arange(n_interpolation_points_1d) ** 2 +
                                                   (np.arange(n_interpolation_points_1d) ** 2).reshape(-1,
                                                                                                       1)) * whsquare,
                                            -gamma_matrix)

    circulant_kernel_tilde = np.zeros(
        (n_fft_coeffs, n_fft_coeffs), dtype=np.float32)
    circulant_kernel_tilde[n_interpolation_points_1d:,
    n_interpolation_points_1d:] = half_kernel
    circulant_kernel_tilde[1:n_interpolation_points_1d + 1,
    n_interpolation_points_1d:] = np.flipud(half_kernel)
    circulant_kernel_tilde[n_interpolation_points_1d:,
    1:n_interpolation_points_1d + 1] = np.fliplr(half_kernel)
    circulant_kernel_tilde[1:n_interpolation_points_1d + 1,
    1:n_interpolation_points_1d + 1] = np.fliplr(np.flipud(half_kernel))
    fft_kernel_tilde = rfft2(circulant_kernel_tilde)
    box_idx = np.ndarray((N, 2), dtype=np.int32)

    Box_idx(box_idx, Y, box_width, min_coord, n_boxes_per_dim, N)

    y_in_box = np.zeros_like(Y)

    Y_in_box(y_in_box, Y, box_idx, box_width, min_coord, n_boxes_per_dim, N)

    denominator_sub = (y_tilde_spacings.reshape(-1, 1) - y_tilde_spacings)
    np.fill_diagonal(denominator_sub, 1)
    denominator = denominator_sub.prod(axis=0)

    interpolate_values = np.ndarray(
        (N, n_interpolation_points, 2), dtype=np.float32)
    Interpolate(y_in_box, y_tilde_spacings, denominator,
                interpolate_values, n_interpolation_points, N)

    w_coefficients = np.zeros((n_boxes_per_dim * n_interpolation_points,
                               n_boxes_per_dim * n_interpolation_points, squared_n_terms), dtype=np.float32)

    Compute_w_coeff(w_coefficients, box_idx, ChargesQij, interpolate_values,
                    n_interpolation_points, n_boxes_per_dim, n_terms, N)

    mat_w = np.zeros((2 * n_boxes_per_dim * n_interpolation_points, 2 *
                      n_boxes_per_dim * n_interpolation_points, n_terms), dtype=np.float32)
    mat_w[:n_boxes_per_dim * n_interpolation_points,
    :n_boxes_per_dim * n_interpolation_points] = w_coefficients
    mat_w = mat_w.transpose((2, 0, 1))
    fft_w = rfft2(mat_w)
    rmut = fft_w * fft_kernel_tilde
    output = irfft2(rmut)
    potentialsQij = np.zeros((N, n_terms), dtype=np.float32)
    PotentialsQij(potentialsQij, box_idx, interpolate_values,
                  output, n_interpolation_points, n_boxes_per_dim, n_terms, N)
    neg_f = np.ndarray((N, 2), dtype=np.float32)
    PotentialsCom = potentialsQij[:, 2].reshape((-1, 1))
    PotentialsXY = potentialsQij[:, :2]
    neg_f = PotentialsCom * Y - PotentialsXY
    return neg_f


def ibFFT_CPU_aw(pos, edgesrc, edgetgt, attr_weight, n_interpolation_points=3, intervals_per_integer=1.0,
              min_num_intervals=100,
              alpha=0.3, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    # 使用修改后的AttrForce函数来计算引力
    numba_attr_weight = Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.float64
    )
    for (i, j), coeff in attr_weight.items():
        if isinstance(i, int) and isinstance(j, int) and isinstance(coeff, float):
            numba_attr_weight[(i, j)] = coeff
        else:
            print(i, j, coeff)
            raise TypeError(f"类型错误：i={type(i)}, j={type(j)}, coeff={type(coeff)}")

    # 通过修改过的AttrForce函数计算
    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3

        modified_AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(beta), d3alpha, numba_attr_weight)

        # 继续进行原有的计算过程
        dC += attr_force
        dC -= 0.01 * d3alpha * \
              torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma),
                                        np.float32(paraFactor))
        ApplyForce(dC, pos, N)
        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
        # move to center
        dC *= 0.6
        if ((it + 1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
    print("\n", end="")
    ed = time.time()
    return pos, ed - st


def ibFFT_CPU(pos, edgesrc, edgetgt, n_interpolation_points=3, intervals_per_integer=1.0, min_num_intervals=100,
              alpha=0.1, beta=8, gamma=8, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)
        dC += attr_force
        dC -= 0.01 * d3alpha * \
              torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma),
                                        np.float32(paraFactor))
        ApplyForce(dC, pos, N)
        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
        # move to center
        dC *= 0.6
        if ((it + 1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
    print("\n", end="")
    ed = time.time()
    return pos, ed - st


def generate_k_sequence(num_iterations, k_initial=1.0):
    """
    生成一个参数序列，其中每个参数 k_i 根据前一个参数 k_{i-1} 和迭代次数 n 计算，
    公式为 k_i = 1 / (1 + n / 10) * k_{i-1}。

    参数:
    num_iterations (int): 迭代次数。
    k_initial (float): 序列的第一个元素 k_0 的初始值，默认为 1.0。

    返回:
    list: 包含参数 k 的序列。
    """
    k_sequence = [k_initial]  # 初始化序列，包含第一个元素 k_0
    for n in range(1, num_iterations):
        k_prev = k_sequence[-1]  # 获取前一个元素 k_{i-1}
        k_current = 1 / (1 + n / 10) * k_prev  # 根据公式计算当前元素 k_i
        k_sequence.append(k_current)  # 将当前元素添加到序列中
    return k_sequence


def ibFFT_CPU_ring(pos, edgesrc, edgetgt, centers, ranges, attraction_strength, n_interpolation_points=3,
                   intervals_per_integer=1.0, min_num_intervals=100,
                   alpha=0.1, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    decay_rate = 0.5
    bool_break = True
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    k_values = generate_k_sequence(max_iter)
    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)

        for i in range(N):
            current_pos = pos[i]
            center = centers[i]

            # 计算到中心的距离
            distance_to_center = np.sqrt((current_pos[0] - center[0]) ** 2 + (current_pos[1] - center[1]) ** 2)

            # 获取允许的距离范围
            d = ranges[i][0]  # d
            d_minus_1 = ranges[i][1]  # d - 1

            # 根据距离调整吸引力强度
            if distance_to_center > d:
                attraction_strength[i] *= 1 + k_values[it]  # 增加吸引力强度
            elif distance_to_center < d_minus_1:
                attraction_strength[i] *= 1 - k_values[it]  # 减少吸引力强度

        # 为每个点施加吸引力
        ApplyCenterForce(pos, centers, attraction_strength, N)

        dC += attr_force
        dC -= 0.01 * d3alpha * \
              torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma),
                                        np.float32(paraFactor))
        ApplyForce(dC, pos, N)

        for i in range(N):
            # 获取当前节点的位置和中心
            current_pos = pos[i]
            center = centers[i]

            # 检查距离是否超出范围
            distance_to_center = np.sqrt((current_pos[0] - center[0]) ** 2 + (current_pos[1] - center[1]) ** 2)

            # 获取允许的距离范围
            d = ranges[i][0]  # d
            d_minus_1 = ranges[i][1]  # d - 1

            # 检查距离是否超出范围
            if distance_to_center > d or distance_to_center < d_minus_1:
                bool_break = False
                # 计算单位向量（方向）
                direction = (current_pos - center) / (distance_to_center + 1e-12)  # 避免除以零

                # 移动节点到距离中心 d - 0.5 的位置
                new_position = center + direction * (d - 0.5)
                pos[i] = new_position
            if it > max_iter / 2 and bool_break:
                break
        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
        # move to center
        dC *= 0.6
        if ((it + 1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
    print("\n", end="")
    ed = time.time()
    return pos, ed - st, bool_break


def ibFFT_CPU_SUPER(pos, edgesrc, edgetgt, super, ranges, n_interpolation_points=3, intervals_per_integer=1.0,
                    min_num_intervals=100,
                    alpha=0.1, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    gamma_super = 200
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    k_values = generate_k_sequence(max_iter)
    super_dis = True
    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)

        dC += attr_force
        dC -= 0.01 * d3alpha * \
              torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive_super(pos, n_interpolation_points,
                                              intervals_per_integer, min_num_intervals, np.float32(gamma),
                                              np.float32(paraFactor), super)
        ApplyForce(dC, pos, N)
        super_dis = True
        for i in range(len(super)):
            x1, y1 = pos[super[i]]
            R1 = ranges[i]

            for j in range(i + 1, len(super)):  # j > i
                x2, y2 = pos[j]
                R2 = ranges[j]

                # 计算两点之间的距离
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # 如果两点之间的距离小于 Ri + Rj，则需要调整
                if dist < R1 + R2:
                    super_dis = False
                    # 计算两点的连线方向向量
                    direction = np.array([x2 - x1, y2 - y1])
                    # 计算需要移动的距离
                    move_distance = (R1 + R2 - dist) / 2  # 每个点需要移动的距离

                    # 单位化方向向量
                    direction_norm = np.linalg.norm(direction)
                    unit_direction = direction / direction_norm

                    # 更新坐标，将两点向反方向移动
                    pos[i] = (x1 - move_distance * unit_direction[0], y1 - move_distance * unit_direction[1])
                    pos[j] = (x2 + move_distance * unit_direction[0], y2 + move_distance * unit_direction[1])
                    ranges[i] = ranges[i]*0.99
                    ranges[j] = ranges[j]*0.99
        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
        # move to center
        dC *= 0.6
        if ((it + 1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
        if super_dis == True and it > max_iter * 2 / 3: break
    print("\n", end="")
    print(super_dis)
    ed = time.time()
    return pos, ed - st


def ibFFT_CPU_TN(pos, edgesrc, edgetgt, centers, ranges, n_interpolation_points=3, intervals_per_integer=1.0,
                 min_num_intervals=100,
                 alpha=0.1, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    k_values = generate_k_sequence(max_iter)
    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)

        dC += attr_force
        dC -= 0.01 * d3alpha * \
              torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma),
                                        np.float32(paraFactor))
        ApplyForce(dC, pos, N)

        # for i in range(N):
        #     if ranges[i] == 0:
        #         continue
        #     current_pos = pos[i]
        #     center = centers[i]
        #
        #     # 计算到中心的距离
        #     distance_to_center = np.sqrt((current_pos[0] - center[0]) ** 2 + (current_pos[1] - center[1]) ** 2)
        #
        #     # 获取允许的距离范围
        #     d = ranges[i]
        #
        #     # 调整超界节点位置
        #     if distance_to_center > d:
        #         pos[i][1] = (current_pos[1] - center[1]) * d / distance_to_center
        #         pos[i][0] = (current_pos[0] - center[0]) * d / distance_to_center

        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
        # move to center
        dC *= 0.6
        if ((it + 1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
    print("\n", end="")
    ed = time.time()
    return pos, ed - st

def ibFFT_CPU_A(pos, edgesrc, edgetgt,super, centers, ranges, n_interpolation_points=3, intervals_per_integer=1.0,
                 min_num_intervals=100,
                 alpha=0.1, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    if (alpha != 0):
        paraFactor /= alpha
        # for keeping a same long range force
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    k_values = generate_k_sequence(max_iter)
    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(
            beta), d3alpha)

        dC += attr_force
        dC -= 0.01 * d3alpha * \
              torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma),
                                        np.float32(paraFactor))
        centers = np.array(centers)
        ranges = np.array(ranges)
        super = np.array(super)
        ApplyForce_align(dC,super, pos, N)
        ApplyRepulsiveCenterForce(pos, centers, ranges, super, N)

        # 1 - pow(0.02, 1 / 300) = 0.012955423246736264
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264
        # pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        # pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2

        # 计算所有节点的 x 和 y 坐标的中心点
        center_x = (pos[:, 0].max() + pos[:, 0].min()) / 2
        center_y = (pos[:, 1].max() + pos[:, 1].min()) / 2
        # 获取不属于 super 的节点的索引
        all_indices = np.arange(len(pos))
        not_in_super_indices = np.setdiff1d(all_indices, list(super))

        # 只移动不属于 super 的节点
        pos[not_in_super_indices, 0] -= center_x
        pos[not_in_super_indices, 1] -= center_y
        # move to center
        dC *= 0.6
        if ((it + 1) % 5 == 0):
            print(".", end="")
            sys.stdout.flush()
    print("\n", end="")
    ed = time.time()
    return pos, ed - st

def ibFFT_CPU_TN_multi(pos, edgesrc, edgetgt, centers, ranges, n_interpolation_points=3, intervals_per_integer=1.0,
                       min_num_intervals=100,
                       alpha=0.1, beta=8, gamma=2, max_iter=300, combine=True, seed=None):
    paraFactor = 1.0
    if alpha != 0:
        paraFactor /= alpha
    d3alpha = 1.0
    d3alphaMin = 0.01
    E = edgetgt.shape[0]
    N = len(pos)
    if E / N >= 15.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0  # a small d3alpha for higher average degrees
    if E / N >= 50.0:
        d3alpha /= 10.0
        d3alphaMin /= 10.0
    pos = np.array(pos, dtype=np.float32)
    st = time.time()
    dC = np.zeros((N, 2), dtype=np.float32)
    attr_force = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
    pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2
    edgesrc = np.array(edgesrc, dtype=np.int32)
    edgetgt = np.array(edgetgt, dtype=np.int32)
    bias = np.zeros(edgetgt.shape[0], dtype=np.float32)
    computebias(N, edgesrc, edgetgt, bias)
    if seed is not None:
        torch.manual_seed(seed)
    k_values = generate_k_sequence(max_iter)

    for it in range(max_iter):
        if combine:
            if it == (18 * max_iter // 20):
                n_interpolation_points = 2
            if it == (19 * max_iter // 20):
                n_interpolation_points = 3

        # 计算吸引力
        AttrForce(attr_force, dC, pos, edgesrc, edgetgt, bias, N, np.float32(beta), d3alpha)

        # 排斥力及随机扰动
        dC += attr_force
        dC -= 0.01 * d3alpha * torch.normal(0, 1, size=pos.shape).numpy().astype(np.float32)
        dC += d3alpha * ibFFT_repulsive(pos, n_interpolation_points,
                                        intervals_per_integer, min_num_intervals, np.float32(gamma),
                                        np.float32(paraFactor))
        ApplyForce(dC, pos, N)

        # 约束节点到其自身的 range 和 center
        for i in range(N):
            if ranges[i] == 0:
                continue
            current_pos = pos[i]
            center = centers[i]

            # 计算到中心的距离
            distance_to_center = np.sqrt((current_pos[0] - center[0]) ** 2 + (current_pos[1] - center[1]) ** 2)

            # 获取允许的距离范围
            d = ranges[i]

            # 调整超界节点位置
            if distance_to_center > d:
                pos[i][1] = center[1] + (current_pos[1] - center[1]) * d / distance_to_center
                pos[i][0] = center[0] + (current_pos[0] - center[0]) * d / distance_to_center

        # 限制 ranges[i] == 0 的节点不进入任何 range 的范围
        for i in range(N):
            if ranges[i] == 0:
                current_pos = pos[i]
                for j in range(N):
                    if i == j or ranges[j] == 0:
                        continue
                    center_j = centers[j]
                    range_j = ranges[j]

                    # 检查当前节点是否进入 range_j 的范围
                    distance_to_j = np.sqrt((current_pos[0] - center_j[0]) ** 2 + (current_pos[1] - center_j[1]) ** 2)
                    if distance_to_j < range_j:
                        # 将节点推离 range_j 的边界
                        pos[i][1] = center_j[1] + (current_pos[1] - center_j[1]) * range_j / distance_to_j
                        pos[i][0] = center_j[0] + (current_pos[0] - center_j[0]) * range_j / distance_to_j

        # 更新 d3alpha
        d3alpha += (d3alphaMin - d3alpha) * 0.012955423246736264

        # 调整整个布局到中心
        pos[:, 0] -= (pos[:, 0].max() + pos[:, 0].min()) / 2
        pos[:, 1] -= (pos[:, 1].max() + pos[:, 1].min()) / 2

        dC *= 0.6

        if (it + 1) % 5 == 0:
            print(".", end="")
            sys.stdout.flush()

    print("\n", end="")
    ed = time.time()
    return pos, ed - st
