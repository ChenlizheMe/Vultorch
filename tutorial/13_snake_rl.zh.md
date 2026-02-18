# 13 — 贪吃蛇 RL

!!! tip "不只是神经渲染"
    Vultorch 不只能做 NeRF 和 3DGS。任何你能写进 `torch.Tensor` 的东西
    ——强化学习、物理仿真、信号处理——都能用零拷贝 GPU 张量以 60 fps 可视化。

## 本章新朋友

| 名称 | 作用 | 类比 |
|------|------|------|
| **Snake 环境** | 基于格子的游戏，纯 Python + deque | OpenAI Gym，但只有 50 行 |
| **DQN 智能体** | 3 层 MLP，把观测映射到 Q 值 | 最简单的深度 RL 算法 |
| **经验回放** | 存储转换，随机抽样批次 | 一个让 RL 稳定的 `deque` |
| **ε-贪心** | 以衰减概率随机行动 | 探索 vs 利用 |
| **Q 值热力图** | 颜色编码的格子，展示智能体的"思考" | 类似注意力图，但用于 RL |
| **手动模式** | 用按钮亲自操控蛇 | 自己玩一把来调试环境 |

---

## 为什么选贪吃蛇？

贪吃蛇是完美的 RL 演示：

- **规则简单** ——即使非 ML 人士也能秒懂
- **视觉反馈** ——你能在 32×16 的棋盘上 *看到* 智能体变聪明
- **训练快速** ——DQN 大约 2000 局就能学会基本觅食
- **代码紧凑** ——环境 + 智能体 + 训练 + 可视化，一个文件搞定

而且它证明了一个重要观点：**Vultorch 是通用的 GPU 可视化工具**，
不只是神经渲染库。

---

## 环境——50 行纯 Python

```python
class SnakeEnv:
    def reset(self):
        self.snake = deque([(ROWS // 2, COLS // 2)])
        self.dir = RIGHT
        self._place_food()
        ...
        return self._obs()

    def step(self, action):
        # action: 0=直走, 1=左转, 2=右转
        ...
        return obs, reward, done
```

观测是一个 11 维向量：

| 维度 | 含义 |
|------|------|
| 0–2 | 前方 / 左方 / 右方 是否有危险（墙壁或身体） |
| 3–6 | 当前方向（one-hot 编码） |
| 7–10 | 食物方向（上 / 右 / 下 / 左） |

这对一个小 MLP 来说已经足够了。不需要像素输入，不需要 CNN——
只要最基本的空间关系。

---

## DQN 智能体——3 行 PyTorch

```python
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, 3),  # 直走, 左转, 右转
        )

    def forward(self, x):
        return self.net(x)
```

三个动作（直走 / 左转 / 右转）而不是四个（上下左右）——这让学习
简单得多，因为智能体不需要学习"不要倒退咬自己"。

---

## 经验回放与训练

```python
class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        ...
```

DQN 训练步骤是教科书式的：

```python
def train_dqn():
    s, a, r, s2, d = buf.sample(BATCH)
    with torch.no_grad():
        q2 = target(s2).max(dim=1).values
        y = r + GAMMA * q2 * (1 - d)
    q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad(); loss.backward(); opt.step()
```

目标网络（每 500 步更新一次）稳定学习过程。
ε 从 1.0 衰减到 0.01——智能体从完全随机开始，
逐渐转向利用学到的知识。

---

## 渲染游戏面板

```python
def render_game():
    board = torch.zeros(ROWS, COLS, 3)
    board[:, :] = torch.tensor([0.05, 0.05, 0.08])  # 暗色背景

    # 蛇身（从亮到暗的绿色渐变）
    for i, (r, c) in enumerate(env.snake):
        if i == 0:
            board[r, c] = torch.tensor([1.0, 0.9, 0.0])  # 蛇头：黄色
        else:
            t = 1.0 - i / max(len(env.snake), 1) * 0.5
            board[r, c] = torch.tensor([0.0, t, 0.2])

    # 食物：红色
    board[fr, fc] = torch.tensor([1.0, 0.15, 0.15])

    game_t.copy_(cpu_rgba.to(disp_dev))
```

网格是 32×16（2:1 横屏比例）。画布上的 `filter="nearest"` 设置防止模糊——每个格子都是清晰的像素，
和 Conway 生命游戏的例子如出一辙。

---

## Q 值热力图

Q 值面板可视化了智能体的决策：

- **绿色** = 智能体想直走
- **蓝色** = 智能体想左转
- **橙色** = 智能体想右转
- 左上角 3 个格子显示每个动作的 Q 值大小

这在功能上类似于 Transformer 中的注意力图——它展示的是
*模型在想什么*，而不只是它做了什么。

---

## 手动模式

勾选 **Manual Mode** 复选框，三个按钮就会出现：Left、Fwd、Right。
现在 *你* 来操控贪吃蛇。这对调试 RL 环境非常有用——如果你自己都
解不了这个游戏，智能体大概也不行。

---

## 训练循环

```python
while view.step():
    for _ in range(S["speed"]):
        # ε-贪心动作选择
        if random.random() < S["eps"]:
            a = random.randint(0, 2)
        else:
            a = policy(obs).argmax().item()

        next_obs, reward, done = env.step(a)
        buf.push(obs, a, reward, next_obs, float(done))
        train_dqn()

        if done:
            obs = env.reset()

    render_game()
    render_qvalues()
    view.end_step()
```

**Steps/Frame** 滑块控制每帧执行多少个环境步骤。
调到 50 加速训练；设为 1 慢动作观察每一步。

---

## 完整代码

```python title="examples/13_snake_rl.py"
--8<-- "examples/13_snake_rl.py"
```

---

## 刚才发生了什么？

在一个 Python 文件里你构建了：

1. 一个带 **奖励塑形的贪吃蛇游戏环境**
2. 一个带经验回放和目标网络的 **DQN 智能体**
3. **实时可视化** ——游戏面板 + Q 值热力图 + 奖励曲线
4. **手动模式** ——亲自玩游戏来调试环境
5. **ε-贪心探索** 并实时显示 epsilon 值

全部通过 Vultorch 的零拷贝张量显示以 60 fps 渲染。不需要
单独的 Gym 渲染器，不需要 matplotlib，不需要日志写盘。

**要点：只要你的数据在张量里，Vultorch 就能可视化——
无论是神经辐射场、高斯溅射、细胞自动机，还是一条学习觅食的蛇。**

---

## 关键收获

| 概念 | 代码 | 用途 |
|------|------|------|
| Snake 环境 | `SnakeEnv` 类 | 32×16 网格 RL 环境 |
| DQN | `nn.Sequential(11 → 128 → 128 → 3)` | 深度 Q 网络 |
| 经验回放 | `deque(maxlen=50000)` | 稳定的离策略学习 |
| ε-贪心 | `random() < eps` | 探索 vs 利用 |
| 最近邻过滤 | `filter="nearest"` | 像素级格子显示 |
| Q 值热力图 | 颜色编码的智能体决策 | 视觉调试 |
| 手动模式 | `checkbox("Manual Mode")` | 调试环境 |
| step()/end_step() | 训练循环控制渲染 | RL 循环掌控节奏 |
