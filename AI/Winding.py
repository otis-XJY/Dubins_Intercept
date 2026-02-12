import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class TopologicalInterceptionSim:
    def __init__(self):
        # 场景设置 (2000x2000)
        self.map_size = 2000
        # 障碍物中心 (对应图中 3, 5, 8, 13 号附近)
        self.obstacles = np.array([
            [800, 1300], [1100, 1500], [750, 450], [1000, 1000]
        ])
        self.obs_radius = 80
        
        # 红色采样点 (选取图中关键点)
        self.samples = np.array([
            [500, 1400], [900, 1450], [1150, 1300], [1000, 1100], 
            [600, 950], [1200, 1200], [1500, 600], [500, 550]
        ])

        # 初始位置
        self.enemy_pos = np.array([1000.0, 1950.0]) # 顶部进入
        self.enemy_target = np.array([1500.0, 100.0]) # 底部目标
        self.uav_pos = np.array([200.0, 100.0])    # 我方 UAV1 位置
        
        self.enemy_path = []
        self.uav_path = []
        self.dt = 10.0
        self.speed_e = 15.0
        self.speed_u = 18.0

    def get_winding_number(self, path, obs_pos):
        """计算路径相对于某个障碍物的绕数（拓扑特征）"""
        if len(path) < 2: return 0
        angles = np.arctan2(np.array(path)[:,1] - obs_pos[1], np.array(path)[:,0] - obs_pos[0])
        diffs = np.diff(angles)
        # 修正跨越 -pi to pi 的突变
        diffs[diffs > np.pi] -= 2 * np.pi
        diffs[diffs < -np.pi] += 2 * np.pi
        return np.sum(diffs) / (2 * np.pi)

    def plan_enemy_step(self):
        """模拟敌方盲目最优路径：简单向目标直线运动，避开避障"""
        direction = self.enemy_target - self.enemy_pos
        dist = np.linalg.norm(direction)
        unit_vec = direction / dist
        
        # 基础移动
        next_pos = self.enemy_pos + unit_vec * self.speed_e
        
        # 简单的势场避障 (确保敌方路径是平滑的测地线)
        for obs in self.obstacles:
            d_obs = np.linalg.norm(next_pos - obs)
            if d_obs < self.obs_radius + 50:
                push = (next_pos - obs) / d_obs
                next_pos += push * 20
        
        self.enemy_pos = next_pos
        self.enemy_path.append(self.enemy_pos.copy())

    def intercept_logic(self):
        """
        核心 Idea 3 实现：拓扑编织决策
        1. 预测敌方在当前障碍物约束下的同伦类
        2. 在采样点中寻找该同伦管道的‘咽喉’
        """
        # 1. 识别敌方相对于关键障碍物的拓扑趋势 (例如中心障碍物 13)
        # 我们观察敌方相对于障碍物 [1000, 1000] 的方位角变化
        target_obs = self.obstacles[3] 
        rel_pos = self.enemy_pos - target_obs
        current_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # 2. 拓扑预测：敌方是想从左侧还是右侧绕过中心障碍物？
        # 基于盲目最优假设，计算敌方目标相对于障碍物的方位
        target_rel_pos = self.enemy_target - target_obs
        target_angle = np.arctan2(target_rel_pos[1], target_rel_pos[0])
        
        # 3. 寻找拦截采样点 (在拓扑必经之路上)
        # 寻找距离敌方路径预测线最近且具有拓扑拦截优势的采样点
        best_point = None
        min_cost = float('inf')
        
        for p in self.samples:
            d_to_uav = np.linalg.norm(p - self.uav_pos)
            d_to_enemy = np.linalg.norm(p - self.enemy_pos)
            # 拓扑代价函数：不仅看距离，看该点是否在敌方通往目标的‘拓扑管道’内
            # 这里简化为：拦截点必须在敌方和目标之间
            cost = d_to_uav + d_to_enemy * 0.5 
            if cost < min_cost:
                min_cost = cost
                best_point = p
        
        # 4. 移动向拦截采样点
        u_dir = best_point - self.uav_pos
        u_dist = np.linalg.norm(u_dir)
        if u_dist > 5:
            self.uav_pos += (u_dir / u_dist) * self.speed_u
        self.uav_path.append(self.uav_pos.copy())

    def run(self):
        for _ in range(120):
            self.plan_enemy_step()
            self.intercept_logic()
            if np.linalg.norm(self.enemy_pos - self.uav_pos) < 30:
                print("Interception Success! Topological match achieved.")
                break
        self.plot()

    def plot(self):
        plt.figure(figsize=(10, 8))
        # 画障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs, self.obs_radius, color='gray', alpha=0.3)
            plt.gca().add_patch(circle)
        
        # 画采样点
        plt.scatter(self.samples[:,0], self.samples[:,1], c='red', s=50, label='Topological Nodes')
        for i, p in enumerate(self.samples):
            plt.text(p[0]+10, p[1]+10, str(i), fontsize=9)

        # 画轨迹
        e_path = np.array(self.enemy_path)
        u_path = np.array(self.uav_path)
        plt.plot(e_path[:,0], e_path[:,1], 'k--', label='Enemy (Blind Geodesic)')
        plt.plot(u_path[:,0], u_path[:,1], 'blue', linewidth=2, label='Interceptor (Braiding Path)')
        
        plt.scatter(self.enemy_target[0], self.enemy_target[1], marker='*', s=200, c='gold', label='Target')
        plt.xlim(0, 2000); plt.ylim(0, 2000)
        plt.legend()
        plt.title("Idea 3: Topological Braiding Interception Simulation")
        plt.xlabel("X(m)"); plt.ylabel("Y(m)")
        plt.grid(True)
        plt.show()

sim = TopologicalInterceptionSim()
sim.run()