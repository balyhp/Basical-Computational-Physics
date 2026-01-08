import math
import os
import matplotlib.pyplot as plt

# --------- 简单 FFT（ Cooley–Tukey） ---------
def bit_reverse(value, bits):
    """
    Cooley-Tukey method
    """
    r = 0
    for i in range(bits):
        if value & (1 << i):
            r |= 1 << (bits - 1 - i)
    return r

def fft(signal, inverse=False):
    """
    FFT method using bit-reverse
    """
    n = len(signal)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("FFT length must be a power of two.")
    levels = n.bit_length() - 1
    a = [0j] * n
    for i in range(n):
        a[bit_reverse(i, levels)] = signal[i]
    size = 2
    while size <= n:
        ang = 2.0 * math.pi / size * (-1.0 if not inverse else 1.0)
        wm = complex(math.cos(ang), math.sin(ang))
        half = size // 2
        for start in range(0, n, size):
            w = 1.0 + 0j
            for k in range(start, start + half):
                u = a[k]
                t = w * a[k + half]
                a[k] = u + t
                a[k + half] = u - t
                w *= wm
        size <<= 1
    if inverse:
        inv_n = 1.0 / n
        a = [v * inv_n for v in a]
    return a

def next_pow2(n):
    """
    找到大于n的最近的2次幂值
    """
    return 1 << (n - 1).bit_length()

# --------- 数据读取 ---------
def read_sunspots(filepath):
    months = []
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                m = int(float(parts[0]))
                v = float(parts[1])
            except ValueError:
                continue
            months.append(m)
            values.append(v)
    return months, values

# --------- 功率谱计算 ---------
def power_spectrum(values, remove_mean=True):
    """
    compute |c_k|^2 with FFT, 
    return P = [|c_k|^2]
    """
    N = len(values)
    if remove_mean:
        mean = sum(values) / N
    else:
        mean = 0.0
    x = [complex(v - mean, 0.0) for v in values]
    # 零填充到 2 的幂, 方便 FFT
    L = next_pow2(N)
    if L > N:
        x = x + [0j] * (L - N)
    # FFT
    X = fft(x, inverse=False)
    # 标准化（按 1/L）以便不同长度可比；功率为 |c_k|^2
    scale = 1.0 / L
    P = []
    for k in range(L // 2 + 1):  # 只取非负频率
        re = (X[k].real * scale)
        im = (X[k].imag * scale)
        P.append(re * re + im * im)
    return P, L, mean

# --------- 寻找主峰并估计周期（样本单位：月） ---------
def find_peak(P, L):
    """
    skip k=0 part, 在 [1, L/2) index find max P[k]
    return 
        peak: k_max
        period (month) : L/k_max
    """
    kmax = 1
    vmax = P[1] if len(P) > 1 else 0.0
    for k in range(2, len(P) - 1):
        if P[k] > vmax:
            vmax = P[k]
            kmax = k
    if kmax == 0:
        return 0, float("inf")
    period_months = L / kmax
    return kmax, period_months

def plot_spectrum(ks, P, k_peak=None, period_months=None):
    plt.figure(figsize=(8, 3.6), dpi=140)
    plt.plot(ks, P, lw=1.0, color="#1f77b4")
    if k_peak is not None and 0 <= k_peak < len(ks):
        ymax = max(P) if P else 1.0
        plt.axvline(k_peak, color="crimson", ls="--", lw=1.0, alpha=0.9)
        label = f"peak k={k_peak}"
        if period_months and period_months != float('inf'):
            label += f"\nT≈{period_months:.1f} mo ({period_months/12.0:.2f} yr)"
        plt.text(k_peak, 0.85 * ymax, label, color="crimson")
    plt.xlabel("k")
    plt.xlim(0,200)
    plt.ylabel(r"$|c_k|^2$")
    plt.title("Sunspots Power Spectrum")
    plt.tight_layout()
    plt.savefig("./spectrum.png",dpi=300)

def main():
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "sunspots.txt")
    months, values = read_sunspots(data_path)
    if not values:
        print("未读到数据，请确认 sunspots.txt 与本脚本同目录，且每行两列（月份 数量）。")
        return

    # 功率谱（去均值）
    P, L, mean_val = power_spectrum(values, remove_mean=True)
    ks = list(range(len(P)))  # 0..L/2
    k_peak, period_months = find_peak(P, L)

    print(f"N={len(values)},  FFT length L={L},  mean value≈{mean_val:.4f}")
    print(f"Peak k={k_peak}, T≈{period_months:.2f} months ≈{period_months/12.0:.2f} years")
    plot_spectrum(ks, P, k_peak, period_months)

if __name__ == "__main__":
    main()