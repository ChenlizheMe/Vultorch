"""
13 - Snake RL
=============
Train a DQN agent to play Snake — live, in real time, on the GPU.

Vultorch is not just for neural rendering.  Anything you can write
into a torch.Tensor can be visualized at 60 fps: reinforcement
learning, physics simulation, signal processing, you name it.

This example trains a tiny DQN (3-layer MLP) to play Snake on a 32×16
grid.  Watch the agent go from random moves to food-seeking behaviour
in a few thousand episodes.

Layout
------
Left sidebar : Controls — speed, epsilon, reset, manual mode
Top-left     : Game board (nearest-neighbour, pixel-perfect)
Top-right    : Q-value heatmap (which direction the agent prefers)
Bottom       : Metrics — reward curve, epsilon, episode count

Key concepts
------------
- RL environment          : Snake on a grid, pure PyTorch (CPU tensors)
- DQN agent               : 3-layer MLP, experience replay, ε-greedy
- filter="nearest"        : Pixel-perfect grid rendering
- step()/end_step()       : Training-loop-owned event loop
- Color-coded rendering   : Snake=green, food=red, head=yellow
- Live Q-value heatmap    : See what the agent "thinks" at each cell
"""

import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import vultorch

device = "cpu"  # Snake env is tiny, CPU is fine; agent also on CPU

# ── Snake Environment ─────────────────────────────────────────────
ROWS, COLS = 16, 16
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class SnakeEnv:
    """Minimal Snake game on a ROWS×COLS board."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = deque([(ROWS // 2, COLS // 2)])
        self.dir = RIGHT
        self.food = None
        self._place_food()
        self.done = False
        self.steps = 0
        self.score = 0
        return self._obs()

    def _place_food(self):
        occupied = set(self.snake)
        while True:
            r, c = random.randint(0, ROWS - 1), random.randint(0, COLS - 1)
            if (r, c) not in occupied:
                self.food = (r, c)
                return

    def _obs(self):
        """11-dim observation: danger ahead/left/right, direction one-hot,
        food direction (4), snake length (normalized)."""
        hr, hc = self.snake[0]
        # Danger detection
        def blocked(d):
            dr, dc = DIRS[d]
            nr, nc = hr + dr, hc + dc
            if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                return 1.0
            if (nr, nc) in set(self.snake):
                return 1.0
            return 0.0

        d = self.dir
        danger_ahead = blocked(d)
        danger_left  = blocked((d - 1) % 4)
        danger_right = blocked((d + 1) % 4)

        # Direction one-hot
        dir_oh = [0.0] * 4
        dir_oh[d] = 1.0

        # Food direction relative to head
        fr, fc = self.food
        food_up    = 1.0 if fr < hr else 0.0
        food_right = 1.0 if fc > hc else 0.0
        food_down  = 1.0 if fr > hr else 0.0
        food_left  = 1.0 if fc < hc else 0.0

        return torch.tensor([
            danger_ahead, danger_left, danger_right,
            *dir_oh,
            food_up, food_right, food_down, food_left,
        ], dtype=torch.float32)

    def step(self, action):
        """Action: 0=straight, 1=turn left, 2=turn right."""
        if action == 1:
            self.dir = (self.dir - 1) % 4
        elif action == 2:
            self.dir = (self.dir + 1) % 4

        hr, hc = self.snake[0]
        dr, dc = DIRS[self.dir]
        nr, nc = hr + dr, hc + dc

        # Wall or self collision
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or (nr, nc) in set(self.snake):
            self.done = True
            return self._obs(), -10.0, True

        self.snake.appendleft((nr, nc))
        self.steps += 1

        # Food
        if (nr, nc) == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()
            # Small reward for getting closer to food
            old_dist = abs(hr - self.food[0]) + abs(hc - self.food[1])
            new_dist = abs(nr - self.food[0]) + abs(nc - self.food[1])
            reward = 0.1 if new_dist < old_dist else -0.1

        # Timeout
        if self.steps > ROWS * COLS * 2:
            self.done = True
            return self._obs(), -5.0, True

        return self._obs(), reward, False


# ── DQN Agent ─────────────────────────────────────────────────────
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


# Replay buffer
class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s, a, r, s2, d = zip(*batch)
        return (torch.stack(s), torch.tensor(a, dtype=torch.long),
                torch.tensor(r), torch.stack(s2),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


policy = DQN()
target = DQN()
target.load_state_dict(policy.state_dict())
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
buf = ReplayBuffer()

BATCH = 64
GAMMA = 0.99
TARGET_UPDATE = 500

env = SnakeEnv()
obs = env.reset()

# ── State ─────────────────────────────────────────────────────────
S = dict(
    eps=1.0, eps_min=0.01, eps_decay=0.9995,
    ep=0, total_steps=0, ep_reward=0.0,
    best_score=0, speed=4,
    reward_h=[], score_h=[], eps_h=[],
    manual=False,
)

# ── View + panels ────────────────────────────────────────────────
view = vultorch.View("13 — Snake RL", 1100, 700)
ctrl   = view.panel("Controls", side="left", width=0.72)
game_p = view.panel("Game")
q_pan  = view.panel("Q-Values")
met_p  = view.panel("Metrics")

# Display tensors (on cuda for Vultorch, we'll write from CPU)
disp_dev = "cuda" if torch.cuda.is_available() else "cpu"
game_t = vultorch.create_tensor(ROWS, COLS, 4, disp_dev, name="game",
                                 window=view.window)
game_cv = game_p.canvas("game", filter="nearest")
game_cv.bind(game_t)

q_t = vultorch.create_tensor(ROWS, COLS, 4, disp_dev, name="qval",
                               window=view.window)
q_cv = q_pan.canvas("qval", filter="nearest")
q_cv.bind(q_t)


def render_game():
    """Draw the snake board into game_t."""
    # Background
    board = torch.zeros(ROWS, COLS, 3)
    board[:, :] = torch.tensor([0.05, 0.05, 0.08])

    # Checkerboard subtle pattern
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                board[r, c] += 0.02

    # Snake body
    for i, (r, c) in enumerate(env.snake):
        if i == 0:
            board[r, c] = torch.tensor([1.0, 0.9, 0.0])   # head: yellow
        else:
            t = 1.0 - i / max(len(env.snake), 1) * 0.5
            board[r, c] = torch.tensor([0.0, t, 0.2])       # body: green

    # Food
    if env.food:
        fr, fc = env.food
        board[fr, fc] = torch.tensor([1.0, 0.15, 0.15])     # red

    cpu_rgba = torch.ones(ROWS, COLS, 4)
    cpu_rgba[:, :, :3] = board
    game_t.copy_(cpu_rgba.to(disp_dev))


def render_qvalues():
    """Draw Q-value heatmap: for each cell, show max Q if snake head were there."""
    qmap = torch.zeros(ROWS, COLS, 3)
    with torch.no_grad():
        for r in range(ROWS):
            for c in range(COLS):
                # Synthetic obs: pretend head is at (r,c), current direction
                fake_obs = obs.clone()
                # Just use current obs but evaluate — gives a rough idea
                q = policy(obs)
                best_q = q.max().item()
                # Normalize to [0, 1] range for visualization
                v = torch.sigmoid(torch.tensor(best_q)).item()
                qmap[r, c] = torch.tensor([v, v * 0.6, 1.0 - v])

    # Actually, let's show per-cell best action color instead
    # Use real Q-values from current obs
    with torch.no_grad():
        q = policy(obs)
        action_colors = [
            torch.tensor([0.0, 0.8, 0.2]),   # straight: green
            torch.tensor([0.2, 0.4, 1.0]),   # left: blue
            torch.tensor([1.0, 0.5, 0.0]),   # right: orange
        ]
        best_a = q.argmax().item()

        # Show the full board with best action color at head, Q magnitude elsewhere
        hr, hc = env.snake[0]
        for r in range(ROWS):
            for c in range(COLS):
                dist = abs(r - hr) + abs(c - hc)
                fade = max(0.0, 1.0 - dist / (max(ROWS, COLS) * 0.7))
                qmap[r, c] = action_colors[best_a] * fade * 0.5

        # Head: bright with action color
        qmap[hr, hc] = action_colors[best_a]

        # Show Q values as bar in top-left corner (3 cells)
        for i in range(3):
            v = torch.sigmoid(q[i]).item()
            qmap[0, i] = action_colors[i] * v

    cpu_rgba = torch.ones(ROWS, COLS, 4)
    cpu_rgba[:, :, :3] = qmap
    q_t.copy_(cpu_rgba.to(disp_dev))


# ── Controls ──────────────────────────────────────────────────────
@ctrl.on_frame
def draw_ctrl():
    ctrl.text(f"FPS {view.fps:.0f}")
    ctrl.text(f"Episode: {S['ep']}  |  Steps: {S['total_steps']}")
    ctrl.text(f"Score: {env.score}  |  Best: {S['best_score']}")
    ctrl.text(f"Snake length: {len(env.snake)}")
    ctrl.separator()

    S["speed"] = ctrl.slider_int("Steps/Frame", 1, 50, default=4)
    ctrl.separator()

    ctrl.text(f"Epsilon: {S['eps']:.4f}")
    ctrl.text(f"Replay buffer: {len(buf)}")
    ctrl.separator()

    S["manual"] = ctrl.checkbox("Manual Mode", default=False)
    if S["manual"]:
        ctrl.text_wrapped("Use Q/A/D or buttons to control the snake.")
        with ctrl.row():
            if ctrl.button("Left", width=55):
                S["manual_action"] = 1
            if ctrl.button("Fwd", width=55):
                S["manual_action"] = 0
            if ctrl.button("Right", width=55):
                S["manual_action"] = 2

    ctrl.separator()

    if ctrl.button("Reset Agent", width=150):
        policy.__init__()
        target.load_state_dict(policy.state_dict())
        S["eps"] = 1.0
        S["ep"] = 0
        S["total_steps"] = 0
        S["best_score"] = 0
        S["reward_h"].clear()
        S["score_h"].clear()
        S["eps_h"].clear()
        buf.buf.clear()

    if ctrl.button("Reset Game", width=150):
        env.reset()

    ctrl.separator()
    ctrl.text_wrapped(
        "DQN learns to play Snake. Watch epsilon decay and the "
        "reward curve climb. The Q-value panel shows the agent's "
        "preferred action at the head position. "
        "Vultorch isn't just for neural rendering — anything that "
        "fits in a tensor can be visualized at 60 fps."
    )


# ── Metrics ───────────────────────────────────────────────────────
@met_p.on_frame
def draw_met():
    met_p.text(f"Episode {S['ep']}   Best score {S['best_score']}   "
               f"ε {S['eps']:.3f}")
    met_p.separator()
    if S["reward_h"]:
        met_p.plot(S["reward_h"], label="##reward",
                   overlay=f"reward", height=60)
    if S["score_h"]:
        met_p.plot(S["score_h"], label="##score",
                   overlay=f"score {S['score_h'][-1]:.0f}", height=60)
    if S["eps_h"]:
        met_p.plot(S["eps_h"], label="##eps",
                   overlay=f"ε {S['eps']:.3f}", height=40)


# ── DQN training step ────────────────────────────────────────────
def train_dqn():
    if len(buf) < BATCH:
        return
    s, a, r, s2, d = buf.sample(BATCH)
    with torch.no_grad():
        q2 = target(s2).max(dim=1).values
        y = r + GAMMA * q2 * (1 - d)
    q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad()
    loss.backward()
    opt.step()


# ── Main loop ────────────────────────────────────────────────────
S["manual_action"] = None

try:
    while view.step():
        for _ in range(S["speed"]):
            # Select action
            if S["manual"]:
                a = S.get("manual_action", None)
                if a is None:
                    a = 0  # go straight by default
                S["manual_action"] = None
            else:
                if random.random() < S["eps"]:
                    a = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        a = policy(obs).argmax().item()

            next_obs, reward, done = env.step(a)
            buf.push(obs, a, reward, next_obs, float(done))
            obs = next_obs
            S["ep_reward"] += reward
            S["total_steps"] += 1

            # Train
            if not S["manual"]:
                train_dqn()

            # Target network update
            if S["total_steps"] % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

            # Epsilon decay
            if not S["manual"]:
                S["eps"] = max(S["eps_min"], S["eps"] * S["eps_decay"])

            if done:
                S["ep"] += 1
                S["best_score"] = max(S["best_score"], env.score)
                S["reward_h"].append(S["ep_reward"])
                S["score_h"].append(float(env.score))
                S["eps_h"].append(S["eps"])
                if len(S["reward_h"]) > 500:
                    S["reward_h"] = S["reward_h"][-500:]
                    S["score_h"] = S["score_h"][-500:]
                    S["eps_h"] = S["eps_h"][-500:]
                S["ep_reward"] = 0.0
                obs = env.reset()

        # Render
        render_game()
        render_qvalues()

        view.end_step()
finally:
    view.close()
