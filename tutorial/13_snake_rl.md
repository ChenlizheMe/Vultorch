# 13 — Snake RL

!!! tip "Beyond neural rendering"
    Vultorch isn't just for NeRF and 3DGS.  Anything you can write
    into a `torch.Tensor` — reinforcement learning, physics
    simulation, signal processing — can be visualized at 60 fps
    with zero-copy GPU tensors.

## New friends in this chapter

| Name | What it does | Analogy |
|------|-------------|---------|
| **Snake environment** | A grid-based game, pure Python + deque | OpenAI Gym, but 50 lines |
| **DQN agent** | 3-layer MLP that maps observations → Q-values | The simplest deep RL algorithm |
| **Experience replay** | Store transitions, sample random batches | A `deque` that makes RL stable |
| **ε-greedy** | Random moves with decaying probability | Exploration vs exploitation |
| **Q-value heatmap** | Color-coded grid showing the agent's "thinking" | Like an attention map, but for RL |
| **Manual mode** | Take over the snake with buttons | Debug your env by playing it yourself |

---

## Why Snake?

Snake is the perfect RL demo:

- **Simple rules** — even non-ML people understand it instantly
- **Visual feedback** — you can *see* the agent getting smarter on the 32×16 board
- **Fast training** — a DQN learns basic food-seeking in ~2000 episodes
- **Fits in one file** — environment + agent + training + visualization

And it proves an important point: **Vultorch is a general-purpose
GPU visualization tool**, not just a neural rendering library.

---

## The environment — 50 lines of pure Python

```python
class SnakeEnv:
    def reset(self):
        self.snake = deque([(ROWS // 2, COLS // 2)])
        self.dir = RIGHT
        self._place_food()
        ...
        return self._obs()

    def step(self, action):
        # action: 0=straight, 1=turn left, 2=turn right
        ...
        return obs, reward, done
```

The observation is an 11-dimensional vector:

| Dims | Meaning |
|------|---------|
| 0–2 | Danger ahead / left / right (wall or body) |
| 3–6 | Current direction (one-hot) |
| 7–10 | Food direction (up / right / down / left) |

This is enough for a small MLP to learn. No pixels, no CNN — just
the essential spatial relationships.

---

## The DQN agent — 3 lines of PyTorch

```python
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, 3),  # straight, left, right
        )

    def forward(self, x):
        return self.net(x)
```

Three actions (straight / turn left / turn right) instead of four
(up/down/left/right) — this makes learning much easier because the
agent doesn't need to learn "don't reverse into yourself".

---

## Experience replay and training

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

The DQN training step is textbook:

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

A target network (updated every 500 steps) stabilizes learning.
ε decays from 1.0 to 0.01 — the agent starts fully random and
gradually shifts to exploiting what it's learned.

---

## Rendering the game board

```python
def render_game():
    board = torch.zeros(ROWS, COLS, 3)
    board[:, :] = torch.tensor([0.05, 0.05, 0.08])  # dark background

    # Snake body (gradient from bright to dark green)
    for i, (r, c) in enumerate(env.snake):
        if i == 0:
            board[r, c] = torch.tensor([1.0, 0.9, 0.0])  # head: yellow
        else:
            t = 1.0 - i / max(len(env.snake), 1) * 0.5
            board[r, c] = torch.tensor([0.0, t, 0.2])

    # Food: red
    board[fr, fc] = torch.tensor([1.0, 0.15, 0.15])

    game_t.copy_(cpu_rgba.to(disp_dev))
```

The grid is 32×16 (2:1 landscape ratio).  The `filter="nearest"` setting on the canvas prevents blurring —
each grid cell is a crisp pixel, just like the Conway's Game of Life
example.

---

## Q-value heatmap

The Q-value panel visualizes the agent's decisions:

- **Green** = agent wants to go straight
- **Blue** = agent wants to turn left
- **Orange** = agent wants to turn right
- The top-left 3 cells show Q-value magnitude for each action

This is functionally similar to an attention map in a transformer —
it shows you *what the model is thinking*, not just what it does.

---

## Manual mode

Check the **Manual Mode** box and three buttons appear: Left, Fwd,
Right.  Now *you* play the snake.  This is invaluable for debugging
RL environments — if you can't solve the game yourself, the agent
probably can't either.

---

## The training loop

```python
while view.step():
    for _ in range(S["speed"]):
        # ε-greedy action selection
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

The **Steps/Frame** slider controls how many environment steps happen
per rendered frame.  Crank it up to 50 for faster training; set it
to 1 to watch every move in slow motion.

---

## Full code

```python title="examples/13_snake_rl.py"
--8<-- "examples/13_snake_rl.py"
```

---

## What just happened?

In one Python file you built:

1. A **Snake game environment** with reward shaping
2. A **DQN agent** with experience replay and target network
3. **Live visualization** — game board + Q-value heatmap + reward curves
4. **Manual mode** — play the game yourself to debug the env
5. **ε-greedy exploration** with live epsilon display

All rendered at 60 fps via Vultorch's zero-copy tensor display.  No
separate Gym renderer, no matplotlib, no logging to disk.

**The takeaway: if your data lives in a tensor, Vultorch can visualize
it — whether that's neural radiance fields, Gaussian splats, cellular
automata, or a snake learning to find food.**

---

## Key takeaways

| Concept | Code | Purpose |
|---------|------|---------|
| Snake env | `SnakeEnv` class | 32×16 grid RL environment |
| DQN | `nn.Sequential(11 → 128 → 128 → 3)` | Deep Q-Network |
| Replay buffer | `deque(maxlen=50000)` | Stable off-policy learning |
| ε-greedy | `random() < eps` | Exploration vs exploitation |
| Nearest filter | `filter="nearest"` | Pixel-perfect grid display |
| Q-value heatmap | Color-coded agent decisions | Visual debugging |
| Manual mode | `checkbox("Manual Mode")` | Debug the environment |
| step()/end_step() | Training-loop-owned rendering | RL loop controls pace |
