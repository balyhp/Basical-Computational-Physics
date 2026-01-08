import math

class MyRandom:
    """
    线性同余生成器 (LCG)
    用于生成 [0, 1) 之间的均匀分布随机数
    """
    def __init__(self, seed=123456789):
        self.state = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32

    def random(self):
        """返回 [0, 1)"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def uniform(self, a, b):
        """返回 [a, b)"""
        return a + (b - a) * self.random()

class FCCHeisenberg:
    def __init__(self, L, J=1.0):
        self.L = L  # 线性尺寸 (晶胞数量)
        self.J = J
        self.N = 4 * L**3  # FCC 总原子数 = 4 * L^3
        self.spins = [[0.0, 0.0, 1.0] for _ in range(self.N)] # 初始化自旋向上
        self.neighbors = [] # 邻居列表
        self.rng = MyRandom()
        
        self._build_lattice()
        self._randomize_spins()

    def _build_lattice(self):
        """
        构建 FCC 晶格的邻居列表。
        使用整数坐标系: 晶胞边长为 2。
        基原子坐标: (0,0,0), (1,1,0), (1,0,1), (0,1,1)
        """
        # 1. 生成所有站点的坐标
        coords = []
        # 映射坐标到索引 (x,y,z) -> index
        coord_map = {} 
        
        idx = 0
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    # 晶胞基准坐标
                    bx, by, bz = 2*x, 2*y, 2*z
                    # 4个基原子
                    basis = [
                        (bx, by, bz),
                        (bx+1, by+1, bz),
                        (bx+1, by, bz+1),
                        (bx, by+1, bz+1)
                    ]
                    for b in basis:
                        coords.append(b)
                        coord_map[b] = idx
                        idx += 1
        
        # 2. 查找邻居 (距离平方为 2 的点)
        # 12个最近邻的相对位移
        offsets = [
            (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
            (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
            (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
        ]
        
        limit = 2 * self.L # 周期性边界的大小
        
        for i in range(self.N):
            cx, cy, cz = coords[i]
            nn_list = []
            for dx, dy, dz in offsets:
                # 应用周期性边界条件 (PBC)
                nx = (cx + dx) % limit
                ny = (cy + dy) % limit
                nz = (cz + dz) % limit
                
                neighbor_idx = coord_map[(nx, ny, nz)]
                nn_list.append(neighbor_idx)
            self.neighbors.append(nn_list)

    def _randomize_spins(self):
        """初始化随机自旋"""
        for i in range(self.N):
            self.spins[i] = self._random_unit_vector()

    def _random_unit_vector(self):
        """生成球面上均匀分布的单位向量"""
        # 方法: z uniform in [-1, 1], phi uniform in [0, 2pi]
        z = self.rng.uniform(-1.0, 1.0)
        phi = self.rng.uniform(0, 2 * math.pi)
        r_xy = math.sqrt(1.0 - z*z)
        x = r_xy * math.cos(phi)
        y = r_xy * math.sin(phi)
        return [x, y, z]

    def metropolis_step(self, T):
        """执行一步蒙特卡洛扫描 (N次尝试)"""
        beta = 1.0 / T
        
        for _ in range(self.N):
            # 1. 随机选取一个格点
            i = int(self.rng.random() * self.N)
            
            # 2. 计算当前局部场 H_local = sum(S_neighbor)
            hx, hy, hz = 0.0, 0.0, 0.0
            for n_idx in self.neighbors[i]:
                s_n = self.spins[n_idx]
                hx += s_n[0]
                hy += s_n[1]
                hz += s_n[2]
            
            # 3. test一个新的自旋方向
            old_spin = self.spins[i]
            new_spin = self._random_unit_vector()
            
            # 4. 能量差 dE = -J * (S_new - S_old) . H_local
            # E_old = -J * S_old . H_local
            # E_new = -J * S_new . H_local
            dot_old = old_spin[0]*hx + old_spin[1]*hy + old_spin[2]*hz
            dot_new = new_spin[0]*hx + new_spin[1]*hy + new_spin[2]*hz
            
            dE = -self.J * (dot_new - dot_old)
            
            # 5. Metropolis accept/reject
            if dE < 0:
                self.spins[i] = new_spin
            else:
                if self.rng.random() < math.exp(-dE * beta):
                    self.spins[i] = new_spin

    def calculate_properties(self):
        """计算系统的平均能量和磁化强度"""
        E_total = 0.0
        M_vec = [0.0, 0.0, 0.0]
        
        for i in range(self.N):
            sx, sy, sz = self.spins[i]
            M_vec[0] += sx
            M_vec[1] += sy
            M_vec[2] += sz
            
            # 计算能量 (每对相互作用只算一次，所以除以2)
            # 或者遍历所有邻居然后除以2
            local_h_x, local_h_y, local_h_z = 0.0, 0.0, 0.0
            for n_idx in self.neighbors[i]:
                local_h_x += self.spins[n_idx][0]
                local_h_y += self.spins[n_idx][1]
                local_h_z += self.spins[n_idx][2]
            
            E_total += -self.J * (sx*local_h_x + sy*local_h_y + sz*local_h_z)

        E_total /= 2.0 # 修正重复计数
        M_abs = math.sqrt(M_vec[0]**2 + M_vec[1]**2 + M_vec[2]**2)
        return E_total, M_abs

def main():
    L = 4  # 晶格尺寸 (L=4 => N=256) supercell
    # 3D Heisenberg FCC 的 Tc 约为 3.176 (k_B T / J)
    #temperatures = [2.95 ,3.0, 3.05 ,3.1, 3.15 ,3.2, 3.25 ,3.3, 3.35, 3.4, 3.45]
    temperatures = [3.25]
    steps_equil = 2000 # 平衡步数
    steps_meas = 5000  # 测量步数
    
    print(f"Simulating 3D FCC Heisenberg Model (L={L}, N={4*L**3})")
    print(f"{'T':<6} | {'<E>/N':<10} | {'<M>/N':<10} | {'Cv':<10} | {'Chi':<10}")
    print("-" * 56)
    
    sim = FCCHeisenberg(L)
    
    # 降温过程 (Annealing)
    for T in temperatures:
        # 1. 平衡
        for _ in range(steps_equil):
            sim.metropolis_step(T)
            
        # 2. 测量
        E_sum = 0.0
        E_sq_sum = 0.0
        M_sum = 0.0
        M_sq_sum = 0.0
        
        for _ in range(steps_meas):
            sim.metropolis_step(T)
            E, M = sim.calculate_properties()
            E_sum += E
            E_sq_sum += E * E
            M_sum += M
            M_sq_sum += M * M
            
        # 3. 统计平均值
        E_avg = E_sum / steps_meas
        E_sq_avg = E_sq_sum / steps_meas
        M_avg = M_sum / steps_meas
        M_sq_avg = M_sq_sum / steps_meas
        
        # 归一化
        e_per_site = E_avg / sim.N
        m_per_site = M_avg / sim.N
        
        # 热容 Cv = (<E^2> - <E>^2) / (N * T^2)  (k_B=1)
        Cv = (E_sq_avg - E_avg**2) / (sim.N * T * T)
        
        # 磁化率 Chi = (<M^2> - <M>^2) / (N * T)
        Chi = (M_sq_avg - M_avg**2) / (sim.N * T)
        
        print(f"{T:<6.2f} | {e_per_site:<10.4f} | {m_per_site:<10.4f} | {Cv:<10.4f} | {Chi:<10.4f}")

if __name__ == "__main__":
    main()