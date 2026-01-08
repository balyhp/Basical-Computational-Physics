# 1D Kronig-Penney: 平面波基组直接构造 H(k) 并求本征值（无 numpy/scipy）
# V(x)=U0 (0<=x<L_B), 其它 0；周期 a=L_W+L_B
# 基: e^{i(k+G_n)x}, G_n=2πn/a；H_{nm}(k)= (ħ^2/2m)(k+G_n)^2 δ_{nm} + V_{n-m}
# V_m = (U0/a) ∫_0^{L_B} e^{-i G_m x} dx = U0*(L_B/a) (m=0)
# m≠0: U0/a * (1 - e^{-i G_m L_B})/(i G_m) = U0/a * e^{-i G_m L_B/2} * 2 sin(G_m L_B/2)/G_m

import math

# 物理常量
ELECTRON_MASS = 9.10938356e-31
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19

# KP 参数
U0_EV = 2.0
U0 = U0_EV * E_CHARGE
L_W = 0.9e-9
L_B = 0.1e-9
A = L_W + L_B

# 数值参数
NBASIS = 201          # 平面波个数 (奇数：索引 -half..+half)
NUM_BANDS = 6         # 输出的最低能带数
NUM_K = 81            # k 采样：[-π/a, π/a]
JACOBI_TOL = 1e-12
JACOBI_MAX = 200000

# 平面波索引
def pw_indices(M):
    half = M // 2
    return [n - half for n in range(M)]

# 解析 V_m
def V_m(m):
    if m == 0:
        return U0 * (L_B / A)
    Gm = 2.0 * math.pi * m / A
    phi = Gm * L_B * 0.5
    # U0/a * e^{-i Gm L_B/2} * 2 sin(phi)/Gm
    pref = (U0 / A) * (2.0 * math.sin(phi) / Gm)
    return pref * complex(math.cos(-phi), math.sin(-phi))

# 构造 H(k)（复厄米 NBASIS×NBASIS）
def build_H(k):
    idx = pw_indices(NBASIS)
    H = [[0j]*NBASIS for _ in range(NBASIS)]
    # 动能
    coef = (HBAR * HBAR) / (2.0 * ELECTRON_MASS)
    for i, n in enumerate(idx):
        Gn = 2.0 * math.pi * n / A
        H[i][i] = coef * (k + Gn) * (k + Gn)
    # 势耦合
    for i, n in enumerate(idx):
        for j, m in enumerate(idx):
            diff = n - m
            H[i][j] += V_m(diff)
    # 数值厄米化（消除浮点误差）
    for i in range(NBASIS):
        for j in range(i+1, NBASIS):
            hij = 0.5*(H[i][j] + H[j][i].conjugate())
            H[i][j] = hij
            H[j][i] = hij.conjugate()
    return H

# 复厄米 → 2N×2N 实对称嵌入
def hermitian_embed(H):
    M = len(H)
    N = 2*M
    S = [[0.0]*N for _ in range(N)]
    for i in range(M):
        for j in range(M):
            a = H[i][j]
            re = a.real
            im = a.imag
            S[i][j] = re
            S[i][j+M] = -im
            S[i+M][j] = im
            S[i+M][j+M] = re
    return S

# Jacobi 求实对称全部特征值
def jacobi_eigvals(A, tol=JACOBI_TOL, max_iter=JACOBI_MAX):
    n = len(A)
    B = [[A[i][j] for j in range(n)] for i in range(n)]
    def max_off(mat):
        p=q=0; mv=0.0
        for i in range(n):
            row=mat[i]
            for j in range(i+1,n):
                v=abs(row[j])
                if v>mv:
                    mv=v; p=i; q=j
        return p,q,mv
    for _ in range(max_iter):
        p,q,mv = max_off(B)
        if mv < tol:
            break
        app = B[p][p]; aqq = B[q][q]; apq = B[p][q]
        if apq == 0.0:
            continue
        tau = (aqq - app)/(2.0*apq)
        t = math.copysign(1.0,tau)/(abs(tau)+math.sqrt(1.0+tau*tau))
        c = 1.0 / math.sqrt(1.0 + t*t)
        s = t * c
        for k in range(n):
            if k!=p and k!=q:
                bkp = B[k][p]; bkq = B[k][q]
                B[k][p] = c*bkp - s*bkq
                B[p][k] = B[k][p]
                B[k][q] = c*bkq + s*bkp
                B[q][k] = B[k][q]
        bpp = c*c*app - 2*s*c*apq + s*s*aqq
        bqq = s*s*app + 2*s*c*apq + c*c*aqq
        B[p][p]=bpp
        B[q][q]=bqq
        B[p][q]=0.0
        B[q][p]=0.0
    return [B[i][i] for i in range(n)]

# 对单个 k 求最低若干本征值
def solve_k(k):
    H = build_H(k)
    S = hermitian_embed(H)
    vals = jacobi_eigvals(S)
    vals.sort()
    # 嵌入矩阵特征值两两重复（每个原始复特征值出现两次）
    unique = []
    for v in vals:
        if not unique or abs(v - unique[-1]) > 1e-10*(1+abs(unique[-1])):
            unique.append(v)
        if len(unique) >= NUM_BANDS:
            break
    return unique

# k 网格
def k_grid(num):
    return [(-math.pi/A) + (2*math.pi/A)*i/(num-1) for i in range(num)]

def main():
    ks = k_grid(NUM_K)
    band_data = []
    print("计算 k=0 ...")
    e_at_0 = solve_k(0.0)
    print("k=0 处最低能级 (eV):")
    for i,E in enumerate(e_at_0,1):
        print(f"  n={i}: {E/E_CHARGE:.6f}")
    print("扫描能带...")
    for k in ks:
        band_data.append((k, solve_k(k)))
    # 写 CSV
    with open("bands.csv","w",encoding="utf-8") as f:
        header = "k(1/m)," + ",".join(f"E{n}(eV)" for n in range(1,NUM_BANDS+1)) + "\n"
        f.write(header)
        for k, eigs in band_data:
            row = [f"{k:.6e}"] + [f"{E/E_CHARGE:.8f}" for E in eigs]
            f.write(",".join(row)+"\n")
    print("已写出 bands.csv")
    
    import matplotlib.pyplot as plt
    for b in range(NUM_BANDS):
        plt.plot([k*A for k,_ in band_data],
                    [eigs[b]/E_CHARGE for _,eigs in band_data],
                    label=f"Band {b+1}")
    plt.xlabel("k * a")
    plt.ylabel("E (eV)")
    plt.title("Kronig-Penney 能带 (平面波直接对角化)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("bands.png",dpi=160)
    print("已生成 bands.png")

if __name__ == "__main__":
    main()