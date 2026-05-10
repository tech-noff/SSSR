import numpy as np
from itertools import combinations_with_replacement

def build_polynomial_library(A, degree, include_functions=False, include_bias=False, ln_eps=1e-6):
    """
    构造扩展函数库：多项式(至多 degree 阶) + 可选的一元函数 {sin, cos, exp, ln(|.|+eps)}。
    按列拼接返回矩阵 Theta 以及对应的 feature_names（完全对齐）。
    
    参数
    ----
    A : ndarray, shape (N, r)
        每一行是一条样本，每一列是一个变量 z_j
    degree : int
        多项式最高阶 (>=0)。若 include_bias=True 会包含常数项 '1'
    include_functions : bool
        是否额外包含 sin(z_j), cos(z_j), exp(z_j), ln(|z_j|+eps)
    include_bias : bool
        是否包含常数项 '1'
    ln_eps : float
        ln 的安全偏移量，用 ln(|z_j| + ln_eps)

    返回
    ----
    Theta : ndarray, shape (N, n_terms)
        构造出的特征矩阵
    feature_names : list[str], 长度 n_terms
        每一列对应的特征名
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A 必须是二维数组 (N, r)")
    N, r = A.shape
    if degree < 0:
        raise ValueError("degree 必须 >= 0")

    cols = []
    names = []

    # -------- 多项式库 --------
    # 生成所有总阶数 d <= degree 的单项式（用 combinations_with_replacement 枚举指数的“下标多重集”）
    # 约定顺序：先 d=0,1,2,... 其中 d=0 仅在 include_bias=True 时包含
    start_d = 0 if include_bias else 1
    for d in range(start_d, degree + 1):
        for combo in combinations_with_replacement(range(r), d):
            # combo 是长度为 d 的下标多重集，如 (0,0,2) -> z1*z1*z3
            term = np.ones(N, dtype=float)
            if d > 0:
                for idx in combo:
                    term *= A[:, idx]
                name = "*".join([f"z{idx+1}" for idx in combo])
            else:
                name = "1"  # bias
            cols.append(term)
            names.append(name)

    # -------- 一元函数库（仅一次多项式）--------
    if include_functions:
        # sin(z_j)
        for j in range(r):
            cols.append(np.sin(A[:, j]))
            names.append(f"sin(z{j+1})")
        # cos(z_j)
        for j in range(r):
            cols.append(np.cos(A[:, j]))
            names.append(f"cos(z{j+1})")
        # exp(z_j)
        for j in range(r):
            cols.append(np.exp(A[:, j]))
            names.append(f"exp(z{j+1})")
        # ln(|z_j| + eps)
        Aj_safe = np.log(np.abs(A) + ln_eps)
        for j in range(r):
            cols.append(Aj_safe[:, j])
            names.append(f"ln(|z{j+1}|+{ln_eps})")

    # 组装矩阵
    if len(cols) == 0:
        # 如果用户给了 degree=0 且 include_bias=False 且 include_functions=False，库为空
        Theta = np.empty((N, 0), dtype=float)
    else:
        Theta = np.column_stack(cols)

    # 最终检查
    assert Theta.shape[1] == len(names), "列数与特征名数量不一致！"

    return Theta, names