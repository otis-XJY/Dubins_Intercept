#!/usr/bin/env bash
# A/B 实验脚本：比较重构前(b9198a8)与重构后(8c49187)的测试结果
set -euo pipefail

COMMIT_A="b9198a8"  # 重构前
COMMIT_B="8c49187"  # 重构后
TEST_FILES="tests/test_design_d_enhancements.py tests/test_train_online_smoke.py"

echo "===== A/B 实验：对比重构前后测试结果 ====="

run_test() {
    local commit=$1
    local label=$2
    echo ""
    echo ">>> [$label] 切换到 commit $commit"
    git checkout "$commit" 2>&1
    echo ">>> [$label] 运行测试..."
    pytest $TEST_FILES -v --tb=short 2>&1 | tail -50
    echo ">>> [$label] 测试完成"
}

run_test "$COMMIT_A" "A-重构前"
run_test "$COMMIT_B" "B-重构后"

echo ""
echo "===== 恢复到 MARL 分支 ====="
git checkout MARL 2>&1
echo "===== A/B 实验结束 ====="
