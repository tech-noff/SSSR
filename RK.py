def solve_rk4_adaptive_fixed_dt(fun, t0, tf, y0, dt,
                                rtol=1e-6, atol=1e-9,
                                h0=None, h_min=1e-12,
                                safety=0.9, grow=5.0, shrink=0.2,
                                max_steps=10_000_000):
    # --- 基本检查与准备 ---
    length = tf - t0
    N_float = length / dt
    N = int(round(N_float))
    if not np.isclose(N_float, N, atol=1e-12, rtol=0):
        raise ValueError("tf 必须等于 t0 + N*dt（在1e-12容差内）。")
    direction = 1.0 if length >= 0 else -1.0
    if direction < 0:
        dt = -abs(dt)
    else:
        dt = abs(dt)

    y = np.array(y0, dtype=float)
    d = y.size
    T = np.array([t0 + k*dt for k in range(N+1)], dtype=float)
    Y = np.empty((N+1, d), dtype=float)
    Y[0] = y.copy()

    # 初始步长
    if h0 is None:
        h = length / max(100.0, N) if N > 0 else (dt/10.0)
        if h == 0:
            h = dt/10.0
    else:
        h = float(h0)
    h = np.sign(dt) * abs(h)

    p = 4  # RK4阶数
    idx_out = 1
    t = t0

    # --- 逐个输出间隔推进 ---
    steps = 0
    while idx_out <= N:
        t_target = T[idx_out]
        # 在当前输出间隔内用自适应子步推进到 t_target
        while (t - t_target) * direction < 0:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("达到最大步数限制；可能步长过小或系统发散。")

            # 剪裁步长，避免越过本段目标
            if (t + h - t_target) * direction > 0:
                h = t_target - t

            # 一步 h与两步 h/2 的误差估计
            y_full = rk4_step(fun, t, y, h)
            y_half = rk4_step(fun, t, y, 0.5*h)
            y_half2 = rk4_step(fun, t + 0.5*h, y_half, 0.5*h)

            # Richardson 误差估计（阶5）：(y_half2 - y_full)/15
            err_est = (y_half2 - y_full) / 15.0
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_half2))
            err_norm = np.max(np.abs(err_est) / scale)

            if err_norm <= 1.0 or abs(h) <= h_min + 1e-30:
                # 接受步；使用更精确的 y_half2
                t = t + h
                y = y_half2

                # 估计下一步步长
                if err_norm == 0.0:
                    fac = grow
                else:
                    fac = safety * (1.0/err_norm)**(1.0/(p+1))
                    fac = min(grow, max(shrink, fac))
                h = h * fac
                # 不允许跨越目标；若下一步超越，会在下一轮被截断到 (t_target - t)
                if abs(h) < h_min:
                    h = np.sign(h) * h_min
            else:
                # 拒绝步，缩小步长重试
                fac = safety * (1.0/err_norm)**(1.0/(p+1))
                fac = max(shrink, min(1.0, fac))
                h = h * fac
                if abs(h) < h_min:
                    h = np.sign(h) * h_min

        # 到达 t_target（机器精度内）
        Y[idx_out] = y.copy()
        idx_out += 1

    return T, Y

def rk4_step(fun, t, y, h):
    k1 = fun(t, y)
    k2 = fun(t + 0.5*h, y + 0.5*h*k1)
    k3 = fun(t + 0.5*h, y + 0.5*h*k2)
    k4 = fun(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

import re
import numpy as np

# ----------------- 解析 -----------------
_poly_pat = re.compile(r'^z(\d+)(\*z\d+)*$')
_unary_pat = re.compile(r'^(sin|cos|exp|ln)\(\s*([^\)]+)\s*\)$') 
_var_pat   = re.compile(r'z(\d+)')  

def _parse_poly_term(name: str, latent_dim: int):

    parts = name.replace(' ', '').split('*')
    expo = {}
    for p in parts:
        if not p.startswith('z'):
            raise ValueError(f"仅支持多项式形如 'z1*z2'，收到: {name}")
        j = int(p[1:]) - 1
        if j < 0 or j >= latent_dim:
            raise ValueError(f"变量索引越界: {name}（latent_dim={latent_dim}）")
        expo[j] = expo.get(j, 0) + 1
    return sorted(expo.items())

def _parse_unary_term(name: str, latent_dim: int):

    m = _unary_pat.match(name.replace(' ', ''))
    if not m:
        raise ValueError(f"不支持的函数项格式: {name}")
    op = m.group(1)  # sin / cos / exp / ln
    inside = m.group(2)
    m2 = _var_pat.findall(inside)
    if len(m2) != 1:
        raise ValueError(f"一元函数只允许单变量，如 'sin(z3)'；收到: {name}")
    j = int(m2[0]) - 1
    if j < 0 or j >= latent_dim:
        raise ValueError(f"变量索引越界: {name}（latent_dim={latent_dim}）")
    return op, j

def _compile_term(name: str, latent_dim: int):
    """
    将库项编译为描述符：
      - 常数: ('const', c)          # c 为 float
      - 多项式: ('poly', [(j,pow), ...])
      - 一元函数: ('unary', op, j)  其中 op in {'sin','cos','exp','ln'}
    """
    s = name.strip()

    # 1) 先尝试识别“纯数字常数”，例如 '1', '0.5', '-2', '3e-1' 等
    #    注意：这里故意排除包含字母的情况，以免误判 'z1' 之类
    try:
        c = float(s)
        # 不包含字母则认为是常数项
        if not any(ch.isalpha() for ch in s):
            return ('const', c)
    except ValueError:
        pass

    # 2) 多项式（只允许 z1*z2*z2 这种乘积形式）
    if _poly_pat.match(s):
        return ('poly', _parse_poly_term(s, latent_dim))

    # 3) 一元函数 sin(zj) / cos(zj) / exp(zj) / ln(zj)
    if _unary_pat.match(s):
        op, j = _parse_unary_term(s, latent_dim)
        return ('unary', op, j)

    # 4) 其它情况视为非法
    raise ValueError(f"无法识别的库项: {name}")

# ----------------- 构造器 -----------------
def build_f_sparse_mixed(library_names, Supports, coef_matrix, latent_dim, ln_eps=1e-6):

    import numpy as np

    n_eq = len(Supports)
    coef_matrix = np.asarray(coef_matrix, dtype=float)

    if coef_matrix.ndim != 2:
        raise ValueError(f"coef_matrix 应为二维矩阵，当前 shape={coef_matrix.shape}")
    if coef_matrix.shape[0] != n_eq:
        raise ValueError(
            f"coef_matrix 的行数应为 {n_eq} (方程个数)，实际 {coef_matrix.shape[0]}。"
        )

    # 取 Supports 的“去重稳定顺序”索引列表
    used_idx, seen = [], set()
    for row in Supports:
        for k in row:
            if k not in seen:
                seen.add(k)
                used_idx.append(k)

    # 编译这些用到的库项为描述符
    descriptors = []
    for k in used_idx:
        if k < 0 or k >= len(library_names):
            raise ValueError(f"库索引 {k} 越界（library_names 长度={len(library_names)}）")
        descriptors.append(_compile_term(library_names[k], latent_dim))

    # 建立索引映射: 原库索引 -> used_idx 中的位置
    pos_map = {k: i for i, k in enumerate(used_idx)}

    # 预处理每个方程：常数项、(used 位置数组)、(对应系数)
    per_eq = []
    has_explicit_const = None  # 记录使用的是哪种模式

    for i in range(n_eq):
        sup = Supports[i]
        n_sup = len(sup)
        n_cols = coef_matrix.shape[1]

        # 判定本组数据的模式（带常数 / 不带常数）
        if n_cols == 1 + n_sup:
            # 模式 A：第一列是常数项
            c0 = float(coef_matrix[i, 0])
            coef_arr = np.asarray(coef_matrix[i, 1:], dtype=float)
            mode = "with_const"
        elif n_cols == n_sup:
            # 模式 B：无显式常数项
            c0 = 0.0
            coef_arr = np.asarray(coef_matrix[i, :], dtype=float)
            mode = "no_const"
        else:
            raise ValueError(
                f"coef_matrix 第 {i} 行列数不匹配："
                f"应为 {n_sup}（无显式常数）或 {1 + n_sup}（含常数），实际 {n_cols}。"
            )

        # 全部行必须保持同一模式，避免混用
        if has_explicit_const is None:
            has_explicit_const = (mode == "with_const")
        else:
            if has_explicit_const != (mode == "with_const"):
                raise ValueError(
                    "coef_matrix 不同行的列数模式不一致：有的包含显式常数，有的不包含。"
                )

        pos_arr = np.array([pos_map[k] for k in sup], dtype=int)
        per_eq.append((c0, pos_arr, coef_arr))

    # 统计多项式需要的每个变量的最大幂，以及需要的一元函数集合
    max_pow = {}               # var j -> max power
    needed_unary = {'sin': set(), 'cos': set(), 'exp': set(), 'ln': set()}
    for desc in descriptors:
        if desc[0] == 'poly':
            for j, p in desc[1]:
                if p > max_pow.get(j, 0):
                    max_pow[j] = p
        elif desc[0] == 'unary':
            _, op, j = desc
            needed_unary[op].add(j)
        elif desc[0] == 'const':
            # 常数项不依赖 y，不参与 max_pow / needed_unary
            continue
        else:
            raise RuntimeError(f"未知的描述符类型: {desc[0]}")

    # ----------------- f(t,y) -----------------
    def f(t, y):
        y = np.asarray(y, dtype=float)
        if y.shape[0] != latent_dim:
            raise ValueError(f"y.size={y.size} 与 latent_dim={latent_dim} 不符。")

        # 1) 预生成 y 的幂表
        y_pows = {}
        for j, pmax in max_pow.items():
            arr = np.empty(pmax + 1, dtype=float)
            arr[0] = 1.0
            base = y[j]
            for p in range(1, pmax + 1):
                arr[p] = arr[p - 1] * base
            y_pows[j] = arr

        # 2) 预计算需要的一元函数值
        unary_vals = {'sin': {}, 'cos': {}, 'exp': {}, 'ln': {}}
        if needed_unary['sin']:
            for j in needed_unary['sin']:
                unary_vals['sin'][j] = np.sin(y[j])
        if needed_unary['cos']:
            for j in needed_unary['cos']:
                unary_vals['cos'][j] = np.cos(y[j])
        if needed_unary['exp']:
            for j in needed_unary['exp']:
                unary_vals['exp'][j] = np.exp(y[j])
        if needed_unary['ln']:
            for j in needed_unary['ln']:
                unary_vals['ln'][j] = np.log(np.abs(y[j]) + ln_eps)

        # 3) 计算所有 used 库项的数值
        m = len(descriptors)
        vals = np.ones(m, dtype=float)
        for i_term, desc in enumerate(descriptors):
            kind = desc[0]
            if kind == 'poly':
                v = 1.0
                for j, p in desc[1]:
                    if j in y_pows:
                        v *= y_pows[j][p]
                    else:
                        v *= y[j] ** p
                vals[i_term] = v
            elif kind == 'unary':
                _, op, j = desc
                vals[i_term] = unary_vals[op][j]
            elif kind == 'const':
                vals[i_term] = desc[1]  # 直接取常数值
            else:
                raise RuntimeError(f"未知的描述符类型: {kind}")

        # 4) 装配每个方程
        dydt = np.empty(n_eq, dtype=float)
        for i in range(n_eq):
            c0, pos_arr, coef_arr = per_eq[i]
            dydt[i] = c0 + np.dot(coef_arr, vals[pos_arr])
        return dydt

    info = {
        "used_library_indices": used_idx,
        "used_library_names": [library_names[k] for k in used_idx],
        "supports_library_names_per_eq": [[library_names[k] for k in row] for row in Supports],
        "needed_unary": {k: sorted(list(v)) for k, v in needed_unary.items()},
        "max_pow_per_var": max_pow,
        "ln_eps": ln_eps,
        "has_explicit_const": has_explicit_const,
    }
    return f, info