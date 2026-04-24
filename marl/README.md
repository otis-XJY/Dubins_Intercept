# MARL 子包说明

## 推荐目录结构

| 路径 | 职责 |
|------|------|
| `marl/envs/` | Gymnasium 环境（`TODCMARLEnv`，实现见 `envs/todc_env.py`） |
| `marl/obs/` | 观测构造（`TODCObservationGenerator`） |
| `marl/rewards/` | 奖励（`RewardConfig`、`TODCRewardFunction`） |
| `marl/rl/` | 算法（MAPPO：`compute_gae`、`ppo_minibatch_update`） |
| `marl/nn/` | 策略/价值网络与方案构建（`build_actor_critic_schemes` 等） |
| `marl/runners/` | 训练入口（`online_train.py`，`python -m marl.runners.online_train`） |
| `marl/utils/` | 工具（如 `live_server`） |

## 启动训练

```bash
conda activate dubins
cd <项目根目录>
python -m marl.runners.online_train --config configs/train_online0324.yaml
```

## 最小 API 示例

```python
from marl.envs import TODCMARLEnv
from marl.nn.models import build_actor_critic_schemes
from marl.rl.mappo import compute_gae, ppo_minibatch_update
```
