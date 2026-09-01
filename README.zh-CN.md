# onrobot-safe-adaptation

[English](README.md) | **简体中文**

本项目使用 PyTorch 和 Isaac Lab 对 Go2 进行 SQRL 预训练，并通过与
`unitree_mujoco` 及未来实机运行时一致的 SDK2 接口完成 PyTorch 在线微调。
当前 runner 只提供 Isaac Lab 和 MuJoCo 执行入口，没有直接控制真实机器人的命令。

## 快速运行

以下命令都应在仓库根目录执行，并先激活已经安装好依赖的 Python 环境。

### 1. 检查 Python 包导入和最终配置

```bash
python -m pip check
python -c "import torch, mujoco; from train.core.base import ACTION_SPEC; print(ACTION_SPEC.size)"
python -c "import isaaclab, isaacsim; print('Isaac Lab environment is ready')"
python -m train.runner show-config
```

这些命令不会验证真实双 Isaac GPU、DDS 或 X11 运行链路。最后一条命令只解析并
打印预训练使用的 Isaac profile，不会启动 Isaac Sim。使用自定义配置时，可以先
检查合并结果：

```bash
python -m train.runner show-config --config config.example.yaml
```

`show-config` 显示预训练使用的 Isaac profile；`sim` 直接启动外部模拟器，不读取
YAML 配置。

### 2. Isaac Lab 预训练

```bash
python -m train.runner pretrain
```

默认配置创建 512 个 task 环境和 64 个 safety 环境。两组环境分别位于两个
独立的 spawn 子进程中，各自拥有 `AppLauncher`、Isaac 模拟器、episode 状态和
物理时钟。主进程负责策略、QSafe、replay 和严格串行调度。

默认输出：

```text
runs/sqrl/pretrain/final.npz
```

严格暂停需要两个 Isaac/Kit 上下文，并且主进程还持有 PyTorch 模型，因此显存需求
高于单个 Isaac 实例。可以先用小规模配置检查完整流程；这只用于冒烟测试，不代表
论文训练规模。例如新建 `config.smoke.yaml`：

```yaml
algorithm:
  n_pre: 64
  n_target: 64
  n_off: 32
  k: 2
  max_safety_trajectories: 2
  learning_starts: 0
  batch_size: 32
  safety_batch_size: 32

environment:
  nr_envs: 8
  nr_task_envs: 6
  nr_safety_envs: 2
  render: false

runner:
  output_dir: runs/sqrl/smoke-pretrain
  checkpoint_frequency: 10000
```

然后运行：

```bash
python -m train.runner pretrain --config config.smoke.yaml
```

必须满足：

```text
nr_task_envs > 0
nr_safety_envs > 0
nr_task_envs + nr_safety_envs == nr_envs
```

### 3. 准备严格锁步的 unitree_mujoco bridge

仓库中的 Python 训练端不接受原版异步 bridge。先对受支持的
`unitree_mujoco` revision 应用补丁并重新编译：

```bash
cd /path/to/unitree_mujoco
git checkout 4134cb5dc7ff1ba7f484deda48b5274b58694519
git apply /absolute/path/to/ora-refactor/assets/robots/go2/unitree_mujoco_bridge_lockstep.patch
git apply /absolute/path/to/ora-refactor/assets/robots/go2/unitree_mujoco_bridge_lossless_publish.patch
cmake -S simulate -B simulate/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulate/build --parallel
```

完整说明见
[`assets/robots/go2/UNITREE_MUJOCO_BRIDGE.md`](assets/robots/go2/UNITREE_MUJOCO_BRIDGE.md)。
在线 MuJoCo 阶段还要求当前 Python 环境能够导入 `unitree_sdk2py`：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  /path/to/unitree_sdk2_python
python -m pip install -e /path/to/unitree_sdk2_python
python -c "import unitree_sdk2py; print('unitree_sdk2py is ready')"
```

Unitree SDK 固定依赖 CycloneDDS 0.10.2。如果安装时提示无法定位 CycloneDDS，按
[官方安装说明](https://github.com/unitreerobotics/unitree_sdk2_python)编译对应版本，
并设置 `CYCLONEDDS_HOME` 后重新安装。运行 `cmake` 前，还需要先完成
[unitree_mujoco 上游文档](https://github.com/unitreerobotics/unitree_mujoco)
列出的 C++、MuJoCo 和系统依赖安装。

### 4. 启动 MuJoCo 模拟器

终端 A：

```bash
python -m train.runner sim --unitree-root /path/to/unitree_mujoco
```

如果 checkout 位于仓库相邻的 `modules/unitree_mujoco`，可以省略
`--unitree-root`。也可以设置：

```bash
export UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco
python -m train.runner sim
```

`sim` 会设置 `ORSA_STRICT_LOCKSTEP=1`，验证仓库内固定的 Go2 MJCF、执行器、
传感器顺序和 reset 位姿，然后以前台方式运行模拟器。保持终端 A 运行。

### 5. 零样本迁移、在线微调和评估

终端 B：

```bash
# 使用预训练 checkpoint 直接在 MuJoCo 中评估
python -m train.runner zero-shot \
  --checkpoint runs/sqrl/pretrain/final.npz

# 使用预训练 checkpoint 在线微调
python -m train.runner finetune \
  --checkpoint runs/sqrl/pretrain/final.npz

# 评估微调后的 checkpoint
python -m train.runner eval \
  --checkpoint runs/sqrl/finetune/final.npz
```

默认每个 zero-shot 回合最长 10 秒。要让机器人连续运行更久，可只延长评估
回合，例如运行 60 秒：

```bash
python -m train.runner zero-shot \
  --checkpoint runs/sqrl/pretrain/final.npz \
  --episode-seconds 60
```

该参数不会改变训练或 checkpoint 契约；若机器人提前跌倒，安全终止仍会立即复位。

各命令的输入和输出为：

| 命令 | 后端 | 必需 checkpoint | 默认输出 |
|---|---|---|---|
| `pretrain` | Isaac Lab | 无 | `runs/sqrl/pretrain/final.npz` |
| `zero-shot` | MuJoCo/SDK2 | `pretrain` checkpoint | 只输出评估指标 |
| `finetune` | MuJoCo/SDK2 | `pretrain` checkpoint | `runs/sqrl/finetune/final.npz` |
| `eval` | MuJoCo/SDK2 | `finetune` checkpoint | 只输出评估指标 |

如果不使用默认 DDS 设置，模拟器和训练命令必须传入完全相同的 domain 和网络接口：

```bash
# 终端 A
python -m train.runner sim --domain-id 7 --interface eth0

# 终端 B
python -m train.runner zero-shot \
  --domain-id 7 --interface eth0 \
  --checkpoint runs/sqrl/pretrain/final.npz
```

## 项目分层

`train/` 负责后端创建、环境适配、配置和命令调度。完整 SQRL 执行路径，包括
预训练、微调、评估和 checkpoint，都位于 `sqrl/`，并且只依赖标准环境接口。
网络定义保留在 `algorithms/`。
当前 runner 只接受 `sqrl_sac` 算法和 `go2` 环境选择，其他 selector 会直接报错。

```text
sqrl/sac/pytorch/
  workflow.py       公开的阶段、评估和 checkpoint API
  pretrainer.py     Algorithm 1 的 task/safety 协调器
  task_trainer.py   task 环境的 SAC 更新
  safety_trainer.py on-policy 安全轨迹与 QSafe 更新
  finetuner.py      目标域 SQRL 在线微调
  safety_ops.py     安全动作选择和 QSafe target
  replay_buffer.py  transition replay 与更新资格

train/core/
  contracts.py      tensor、物理、reset 和状态契约
  actions.py        归一化动作到关节目标的映射
  observation.py    策略 observation 构造
  reward.py         reward 和 episode 统计
  process_environment.py 同步子进程环境传输
  manifest.py       checkpoint 环境契约
  estimation_*.py   NumPy/Torch 速度估计器

train/isaac/pytorch/
  environment.py    Go2 Isaac 环境
  pretrain_environment.py 独立 task/safety 进程端点
  runtime.py        Isaac manager stepping 和物理帧
  contract.py       tensor 与关节顺序适配
  terrain.py        地形配置
  randomization.py  迁移随机化
  setup.py          Isaac 场景配置

train/mujoco/pytorch/
  environment.py    Go2 MuJoCo 环境
  client.py         SDK2 DDS client
  buffers.py        同步状态 buffer
  messages.py       SDK 消息解码
  mjcf.py           模型契约验证
  reset.py          模拟器 reset controller
  sdk.py            稳定的公开 bridge API
```

## 环境安装

### Isaac Lab、Isaac Sim 和 MuJoCo

项目基础包支持 Python 3.10 或更高版本。完整复现固定使用 Isaac Lab `v2.3.2`、
Isaac Sim `5.1.0`、Python `3.11`，并创建独立 micromamba 环境：

```bash
micromamba create -n onrobot-safe-adaptation python=3.11 pip git -c conda-forge -y
eval "$(micromamba shell hook -s bash)"
micromamba activate onrobot-safe-adaptation

python -m pip install -U pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

export ISAACLAB_ROOT=/path/to/IsaacLab-v2.3.2
git clone --depth 1 --branch v2.3.2 \
  https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
cd "$ISAACLAB_ROOT"
./isaaclab.sh -i

cd /path/to/onrobot-safe-adaptation
python -m pip install -e ".[mujoco]"
```

Isaac Sim 从 NVIDIA Python package index 安装，Python 版本必须与对应 Isaac
Sim release 匹配。

### 仅使用 SDK2/MuJoCo

如果不运行 Isaac 预训练，则不需要安装 Isaac Sim 和 Isaac Lab：

```bash
python -m pip install -e ".[mujoco]"
```

`.[mujoco]` 不包含 `unitree_sdk2py`，仍需按照前面的官方源码方式安装。所有 MuJoCo
命令都需要可见的 X11 窗口、正确设置的 `DISPLAY`，以及系统可加载
`libX11.so.6` 和 `libXtst.so.6`；当前 reset 流程不支持纯 headless 运行。

DDS 传输和模拟器 reset 仍由 host Python 负责；策略推理、安全过滤和在线梯度更新
使用 PyTorch。

## 预训练严格时序、预算和 checkpoint 契约

### 单次训练中的严格时序

双进程不是两套并行训练。唯一的主训练进程严格串行执行：

```text
task.reset
  ↓
task.step → SAC 更新
  ↓
重复 n_off 次
  ↓
safety.reset
  ↓
safety.step，直到获得 k 条完整轨迹
  ↓
QSafe 更新
  ↓
从暂停状态继续 task.step
```

环境 RPC 为阻塞式请求/响应。safety block 期间不会向 task worker 发送任何请求；
task worker 阻塞在 `recv()`，不会执行 `backend.step()` 或 reset。因此 task 的
物理状态、episode 统计、adapter 状态和环境 RNG 都停留在上一次 task step 后的
精确状态。safety block 结束后不会强制 reset task。

这里的“严格暂停”指仿真状态不推进，并不是向操作系统进程发送 `SIGSTOP`，也不依赖
保存/恢复 Isaac 快照。safety block 开始时的 reset 只作用于独立的 safety 实例。

这保证的是 SQRL 算法事件的全序，以及 task Markov 状态链的连续性。task 和
safety 拥有独立物理时钟；它们不是同一个物理 episode 中的两段时间。

worker 异常或超时会终止训练，不会静默重建环境，因为重建会破坏状态连续性。

### 训练预算

本复现把论文的 500,000-step 参考预算解释为 500,000 个进入 task replay 的
transition。使用 512 个 task 环境时，vector step 无法拆分，所以最终为 500,224。
pretrainer 分别报告 task transition 数和 optimizer update 数。默认 task UTD 为
每个新 transition 一次更新，因此修改并行环境数量不会暗中改变优化预算。

每个预训练 block 按 Algorithm 1 的顺序执行：`n_off` 次 task vector interaction，
按照 transition 数累计 SAC update credit；然后在策略不变期间收集 `k` 条完整
轨迹；最后默认执行一次 QSafe 梯度更新。

按照论文第 6 节的 practical implementation，`D_safe` 是一个小型 on-policy
buffer，只保留当前策略快照最近的 `k` 条完整轨迹；下一 block 会原子替换这批
数据。论文第 5.1 节同时提到 replay buffer 中的“策略混合”，与第 6 节存在内部
歧义。本实现优先采用更明确的 practical-implementation 描述，而不会默认积累
off-policy safety 数据。

只有显式配置 `qsafe_epochs_per_block` 时，才启用实验性的 replay-epoch schedule。
论文伪代码把 `n_off` 嵌套在 `n_pre` 中，而 Table 1 把 `n_pre` 描述为预训练总步数。
本实现采用 Table 1 的解释：`n_pre` 是 task transition 总预算，最多超出一个
vector batch。

默认 `n_off=1` 和 `k=nr_safety_envs` 是本项目的 vector scheduler 选择，不是论文
报告的超参数。`learning_starts=5000` 也是本地 SAC 更新延迟；在达到阈值前，task
预训练使用随机动作，而微调阶段始终使用 safety-filtered policy。论文没有给出
这些设置，因此它们都保持可配置。

### 安全动作与 checkpoint

预训练选择 QSafe 值严格小于 `epsilon_safe` 且最接近安全边界的 accepted action，
对应论文第 6 节的安全边界探索。微调和评估使用 Eq. 3 的有限候选 rejection
sampling：从策略独立采样候选，并在 accepted 候选中均匀选择。如果不存在
accepted candidate，则使用风险最小的候选，并记录 fallback。默认候选数为 100；
这是本项目的有限候选近似，不是论文报告的 candidate count，也不是精确解析条件分布。

第 6 节还描述了按原策略密度对保留候选加权，但没有说明 proposal distribution。
对已经来自该策略的候选再次按密度加权会把分布偏向 `pi(a|s)^2`。本实现采用普通
rejection sampling，从而匹配 Eq. 3 的条件分布；这是经过测试的明确选择，但不声称
逐字复现该歧义句子。

保留原始 QSafe 网络，包括最后的 `tanh` 激活。runner 默认每 10,000 个 task
transition 原子保存一个 `step_XXXXXXXXX.npz`，完整结束后保存 `final.npz`；通过
`runner.checkpoint_frequency` 可以修改间隔，设为 `0` 可关闭周期保存。checkpoint
是 policy/QSafe 的跨后端迁移产物，不是包含 optimizer state 的断点续训文件。

NPZ 文件不使用 pickle：JSON 元数据、环境契约和框架无关的 NumPy 权重位于同一个
文件中，Dense kernel 统一采用 Flax 的 `[input, output]` 布局。PyTorch 通过
`SQRLWorkflow.load` 读取；JAX/Flax 可以直接读取同一文件：

```python
from sqrl.checkpoint import flax_params, load_portable_checkpoint

metadata, arrays = load_portable_checkpoint("runs/sqrl/pretrain/final.npz")
policy_variables = flax_params(arrays, "policy")
qsafe_variables = flax_params(arrays, "qsafe")
```

旧版本生成的 version-2 `.model` 仍可由 PyTorch workflow 读取，新文件统一保存为
`.npz`。这里不使用 ONNX，因为 ONNX 是推理图格式，不能完整表达 SQRL 迁移所需的
训练阶段和环境契约。

已回退的 SAC actor 保留原实现的数值近似
`log(1 - tanh(a)^2 + 1e-6)`，用于 tanh change-of-variables 修正，不包装环境
专属动作投影。
关节物理限制和 target-rate 限制属于环境职责。环境报告实际 applied action 作为
诊断信息；replay 保存原始归一化策略命令，使 SAC 和 QSafe 始终在与评分一致的
动作域中训练，而 target projection 属于 `env.step()` 的 transition。

## Isaac Lab 与 MuJoCo 共享契约

两个 adapter 都构造相同的 46 维策略 observation：

```text
[ 0:12] joint_q
[12:24] joint_dq
[24:27] body-frame IMU gyro
[27:30] 共享的 proprioceptive body-velocity estimate
[30:34] 连续 WXYZ quaternion
[34:46] previous applied absolute joint target
```

NumPy 和 Torch 实现使用相同输入进行交叉测试。velocity command 和 simulator
root velocity 都不会进入策略 tensor。

两个 adapter 还共享以下设置：

- 相同 reward 公式；
- 2 ms physics / 20 ms policy cadence；
- 500-step finite horizon；
- terminal-frame reward 和 observation；
- failure 或 timeout 后进行物理 reset；
- 相同稀疏失败规则：roll 或 pitch 超过 0.8 rad，或者局部 base clearance 低于
  0.18 m，并连续持续五个 2 ms frame；
- 标准关节顺序 `FR, FL, RR, RL × hip, thigh, calf`，并验证严格一一映射。

对于 task SAC，500-step `TimeLimit` 是人为 truncation，因此从保存的 reset 前最终
observation bootstrap；只有真实 failure termination 会屏蔽 task target。QSafe
估计完整有限 safety rollout 内的失败概率，所以 termination 或 horizon truncation
都会终止其 Bellman target。这个差异是有意设计的，并且两个后端一致。

默认每条腿关节位姿为 `[0, 0.9, -1.8]`，base origin 位于平地上方 0.289 m。
仓库内 MuJoCo 模型会在普通 `mj_resetData` 后检查四只脚都接触地面。确定性 Isaac
reset 和实时 MuJoCo reset 都会验证：关节位姿误差不超过 1 mrad、base 高度误差
不超过 2 mm、单位 base orientation 误差不超过 1 mrad，并且四个 foot surface
距离地面不超过 3 mm。

Go2 transfer manifest 记录 SDK 关节顺序、共享 reset 位姿、absolute joint-target
语义、PD 增益、torque limit 和共同的 tilt-or-low-base failure label。迁移检查保持
这些语义严格一致，同时允许预期的阶段差异：Isaac 预训练启用 domain
randomization；MuJoCo 微调/评估使用 0.6 m/s 前进目标，而预训练为 0.5 m/s。

两个后端都保持平地、标称摩擦、控制时序、执行器契约、reset 位姿、failure label、
episode horizon 和相同 reward 公式。MuJoCo 还加入较小的 actuator-force sensor
noise，以更接近 torque penalty 所使用的测量路径。reward 为非负的 Gait in Eight
固定前进目标：分段 velocity tracking，加上 yaw-rate、直立姿态和 joint-torque
energy penalty。

manifest 还记录所有策略可见的 velocity-estimator covariance、confidence、
robust-loss 和 innovation-gate 参数。预训练与微调配置只要修改其中任意一项，
checkpoint load 就会拒绝，而不会静默改变 observation 语义。

## 常见运行问题

- **启动两个 Isaac worker 后显存不足：** 降低 `nr_task_envs`、
  `nr_safety_envs` 和 `nr_envs`，但不要把 task/safety 合并到同一个物理时钟；三者
  仍必须满足前述数量关系。
- **worker timeout 或远端 traceback：** 训练会 fail-fast，且不会自动重建 worker。
  先处理 traceback 中的 Isaac、驱动或资源错误，再重新开始训练。
- **`unitree_sdk2py is required`：** `.[mujoco]` 不包含 Unitree Python SDK；按前述
  官方源码步骤安装，并检查 CycloneDDS 0.10.2。
- **无法打开或找到 MuJoCo 窗口：** 确认 `DISPLAY` 指向显示该窗口的 X11 session，
  并安装 `libX11.so.6`、`libXtst.so.6`。当前自动 reset 依赖可见窗口。
- **DDS 无数据或发现错误进程：** 确认模拟器与 learner 的 `--domain-id` 和
  `--interface` 完全相同。
- **CRC、sequence、tick 或 root-truth 错误：** 通常表示使用了未打补丁、revision
  不匹配或非严格锁步的 bridge；重新核对固定 commit、补丁和构建产物。

## MuJoCo bridge 要求与验证范围

所有 MuJoCo 执行，包括 `zero-shot`、`finetune` 和 `eval`，都要求外部 bridge
每次只执行一个 command-driven transaction。仓库补丁会在 policy/learner 工作
期间暂停模拟器；每个带 sequence 的命令只推进十个 2 ms physics step；随后发布
command acknowledgement，并为 root-state truth 加时间戳。

Python adapter 会在写入 replay 前拒绝 learner gap、缺失首帧、混合 command
sequence、错误 LowState CRC，以及缺失或过期的 root truth。`sim` 命令固定使用
仓库内经过验证的 canonical scene，并在启动前检查 actuator、passive joint、
contact friction、reset pose 和 sensor order。由于 learner 无法通过原版 DDS schema
认证任意外部模型，因此不支持自定义 `--scene`。runner 不会修改外部 checkout。

运行仓库契约测试：

```bash
python -B -m unittest discover -s tests -q
```

测试套件会加载仓库内真实 MuJoCo 模型并进行前向计算，同时覆盖算法、双进程暂停、
reset、时序、observation、reward、关节顺序、checkpoint 和分层契约。测试不会
clone 或编译外部 C++ bridge checkout；bridge 文档记录了补丁生成和人工编译验证
所使用的固定 upstream revision。

Isaac 测试是无需完整 Isaac runtime 的 adapter contract test。真实 Isaac
Sim/PhysX smoke test，以及端到端实时 DDS/X11 测试，仍需要对应的外部运行时和硬件。

## 论文

本文档中的 SQRL、章节和公式引用均指：Krishnan Srinivasan、Benjamin Eysenbach、
Sehoon Ha、Jie Tan、Chelsea Finn，
[《Learning to be Safe: Deep RL with a Safety Critic》，arXiv:2010.14603](https://arxiv.org/abs/2010.14603)。
