import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def main():
    # 1. 数据录入 
    T = np.array([0.50, 1.00, 1.50, 2.00, 2.50, 2.80, 2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45, 3.60, 4.00, 4.50])
    Cv = np.array([0.9697, 1.1112, 1.1573, 1.3407, 1.6473, 1.9445, 1.9041, 2.1011, 2.3498, 2.1874, 2.2893, 1.9886, 1.8445, 1.6109, 1.3823, 1.5534, 1.2058, 0.8921, 0.4198, 0.2597])
    Chi = np.array([0.0076, 0.0186, 0.0323, 0.0590, 0.1073, 0.2276, 0.3034, 0.3364, 0.4679, 0.4856, 0.6284, 0.5624, 0.6281, 0.6310, 0.6225, 0.7001, 0.5743, 0.4695,0.2558, 0.1549])

    # 2. 插值处理 (生成平滑曲线)
    T_smooth = np.linspace(T.min(), T.max(), 300)

    # 使用 B-Spline 进行插值 (k=3 表示三次样条)
    spl_Cv = make_interp_spline(T, Cv, k=2)
    Cv_smooth = spl_Cv(T_smooth)

    spl_Chi = make_interp_spline(T, Chi, k=2)
    Chi_smooth = spl_Chi(T_smooth)

    # 3. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # --- 绘制热容 Cv ---
    ax1.set_title('Phase Transition Analysis: 3D FCC Heisenberg Model')
    ax1.plot(T, Cv, 'o', color='blue', label='Simulation Data', markersize=6, alpha=0.7)
    ax1.plot(T_smooth, Cv_smooth, '-', color='blue', label='Spline Interpolation', linewidth=2)
    ax1.set_ylabel('Heat Capacity ($C_v$)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 标记 Cv 的峰值
    cv_peak_idx = np.argmax(Cv_smooth)
    cv_peak_T = T_smooth[cv_peak_idx]
    ax1.axvline(x=cv_peak_T, color='gray', linestyle='--', alpha=0.5)
    ax1.text(cv_peak_T + 0.1, max(Cv_smooth)*0.9, f'Peak ~ {cv_peak_T:.2f}', color='blue')

    # --- 绘制磁化率 Chi ---
    ax2.plot(T, Chi, 's', color='red', label='Simulation Data', markersize=6, alpha=0.7)
    ax2.plot(T_smooth, Chi_smooth, '-', color='red', label='Spline Interpolation', linewidth=2)
    ax2.set_ylabel('Susceptibility ($\chi$)')
    ax2.set_xlabel('Temperature ($T$)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 标记 Chi 的峰值
    chi_peak_idx = np.argmax(Chi_smooth)
    chi_peak_T = T_smooth[chi_peak_idx]
    ax2.axvline(x=chi_peak_T, color='gray', linestyle='--', alpha=0.5)
    ax2.text(chi_peak_T + 0.1, max(Chi_smooth)*0.9, f'Peak ~ {chi_peak_T:.2f}', color='red')

    plt.tight_layout()
    plt.savefig('../pic/phase_transition.png', dpi=300) 

if __name__ == "__main__":
    main()