"""
obtainNeighbour 函数性能测试
对比循环版本和向量化版本的性能差异
"""

import numpy as np
import time
from main.main0319 import obtainNeighbour

def obtainNeighbour_loop_version(PosP, ValuePos, distance):
    """原始循环版本"""
    if PosP.size == 0 or ValuePos.size == 0:
        return [[] for _ in range(len(PosP))]
    
    if distance < 0:
        raise ValueError("距离阈值必须为非负数")
    
    neighbour_list = []
    PosP_2d = PosP[:, :2] if PosP.ndim > 1 and PosP.shape[1] > 2 else PosP.reshape(-1, 2)
    ValuePos_2d = ValuePos[:, :2] if ValuePos.ndim > 1 and ValuePos.shape[1] > 2 else ValuePos.reshape(-1, 2)
    
    for pos in PosP_2d:
        diffs = ValuePos_2d - pos
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        mask = distances <= distance
        neighbours = ValuePos[mask]
        neighbour_list.append(neighbours)
    
    return neighbour_list

def benchmark(func, PosP, ValuePos, distance, iterations=10):
    """运行多次测试取平均时间"""
    times = []
    for _ in range(iterations):
        start = time.time()
        result = func(PosP, ValuePos, distance)
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)

def test_correctness():
    """验证向量化版本与循环版本结果一致"""
    print("=" * 70)
    print("正确性测试：验证向量化版本与循环版本结果一致")
    print("=" * 70)
    
    # 测试数据
    PosP = np.array([[0, 0], [10, 10], [20, 20]])
    ValuePos = np.array([[1, 1], [5, 5], [11, 11], [15, 15], [100, 100]])
    distance = 5.0
    
    result_loop = obtainNeighbour_loop_version(PosP, ValuePos, distance)
    result_vectorized = obtainNeighbour(PosP, ValuePos, distance)
    
    # 比较结果
    all_match = True
    for i, (r_loop, r_vec) in enumerate(zip(result_loop, result_vectorized)):
        if len(r_loop) != len(r_vec):
            print(f"✗ PosP[{i}]: 邻居数量不匹配 (循环版：{len(r_loop)}, 向量化：{len(r_vec)})")
            all_match = False
        elif len(r_loop) > 0 and not np.array_equal(r_loop, r_vec):
            print(f"✗ PosP[{i}]: 邻居坐标不匹配")
            all_match = False
        else:
            print(f"✓ PosP[{i}]: 结果一致 ({len(r_loop)} 个邻居)")
    
    if all_match:
        print("\n✓ 所有测试结果一致！\n")
    else:
        print("\n✗ 测试结果不一致！\n")
    
    return all_match

def test_performance():
    """性能对比测试"""
    print("=" * 70)
    print("性能测试：对比循环版本和向量化版本")
    print("=" * 70)
    
    # 不同规模的测试
    test_cases = [
        ("小规模 (10x50)", 10, 50),
        ("中等规模 (50x200)", 50, 200),
        ("大规模 (100x500)", 100, 500),
        ("超大规模 (200x1000)", 200, 1000),
    ]
    
    distance = 10.0
    iterations = 10
    
    for name, n_pos, n_value in test_cases:
        print(f"\n{name}:")
        print(f"  PosP: {n_pos} 个点，ValuePos: {n_value} 个点")
        
        # 生成随机测试数据
        np.random.seed(42)
        PosP = np.random.rand(n_pos, 2) * 100
        ValuePos = np.random.rand(n_value, 2) * 100
        
        # 测试循环版本
        time_loop, std_loop = benchmark(
            obtainNeighbour_loop_version, 
            PosP, ValuePos, distance, iterations
        )
        
        # 测试向量化版本
        time_vec, std_vec = benchmark(
            obtainNeighbour, 
            PosP, ValuePos, distance, iterations
        )
        
        # 计算加速比
        speedup = time_loop / time_vec if time_vec > 0 else float('inf')
        
        print(f"  循环版本：   {time_loop*1000:.3f} ± {std_loop*1000:.3f} ms")
        print(f"  向量化版本： {time_vec*1000:.3f} ± {std_vec*1000:.3f} ms")
        print(f"  加速比：     {speedup:.2f}x {'⚡' * min(int(speedup), 5)}")

if __name__ == '__main__':
    print("\n" + "🚀" * 35)
    print("obtainNeighbour 函数优化性能测试")
    print("🚀" * 35 + "\n")
    
    # 正确性测试
    if test_correctness():
        # 性能测试
        test_performance()
        
        print("\n" + "=" * 70)
        print("测试完成！✅")
        print("=" * 70)
    else:
        print("\n由于正确性测试失败，跳过性能测试 ❌\n")