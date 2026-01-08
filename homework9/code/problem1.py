import math

class MyRandom:
    """
    (Linear Congruential Generator)
    用于生成 [0, 1) 之间的均匀分布随机数
    公式: X_{n+1} = (a * X_n + c) % m
    """
    def __init__(self, seed=123456789):
        self.state = seed
        # Numerical Recipes (m = 2^32)
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32

    def random(self):
        """返回一个 [0, 1) 区间的浮点数"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

def exact_hypersphere_volume(d):
    """
    d 维单位超球体的精确体积
    公式: V_d = pi^(d/2) / Gamma(d/2 + 1)
    """
    numerator = math.pi ** (d / 2.0)
    denominator = math.gamma(d / 2.0 + 1)
    return numerator / denominator

def monte_carlo_hypersphere_volume(d, num_samples=1000000):
    """
    使用蒙特卡洛方法估算 d 维单位超球体的体积
    原理: 在 [-1, 1]^d 的超立方体中采样，统计落在单位球内的点。
    超立方体体积为 2^d。
    球体积 V_sphere approx V_cube * (count_inside / total_samples)
    """
    rng = MyRandom()
    count_inside = 0

    for _ in range(num_samples):
        sum_sq = 0.0
        for _ in range(d):
            # 生成 [-1, 1] 之间的随机坐标
            # rng.random() 生成 [0, 1)，映射到 [-1, 1) 为: 2 * r - 1
            x = 2.0 * rng.random() - 1.0
            sum_sq += x * x
        
        if sum_sq <= 1.0:
            count_inside += 1

    volume_cube = 2.0 ** d
    estimated_volume = volume_cube * (count_inside / num_samples)
    return estimated_volume

def main():
    dimensions = [2, 3, 4, 5]
    num_samples = 1000000  # 采样点数

    print(f"{'Dim':<5} | {'MC Volume':<12} | {'Exact Volume':<12} | {'Error (%)':<10}")
    print("-" * 46)

    for d in dimensions:
        mc_vol = monte_carlo_hypersphere_volume(d, num_samples)
        exact_vol = exact_hypersphere_volume(d)
        error = (abs(mc_vol - exact_vol) / exact_vol) * 100
        print(f"{d:<5} | {mc_vol:<12.5f} | {exact_vol:<12.5f} | {error:<10.2f}")

if __name__=="__main__":
    main()
    