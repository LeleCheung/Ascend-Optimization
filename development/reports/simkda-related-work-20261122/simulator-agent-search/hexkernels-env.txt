"""Thin gym env for Hexagon kernel authoring. step() = compile+sim (injected gate_fn)
-> tiered reward + bottleneck-driven observation. No heavy imports at module load."""
import json
import os

from hexkernels.gym import reward as R
from hexkernels.gym import bottleneck as B


def load_task_from_disk(task_id, root="data/v6/tasks"):
    d = os.path.join(root, task_id)
    with open(os.path.join(d, "prompt.md"), "r", encoding="utf-8") as f:
        prompt = f.read()
    with open(os.path.join(d, "spec.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)
    return prompt, spec


class HexagonKernelEnv:
    def __init__(self, gate_fn, task_loader=load_task_from_disk, max_turns=3, on_box=False):
        self._gate = gate_fn
        self._load = task_loader
        self.max_turns = max_turns
        self.on_box = on_box
        self._task_id = None
        self._spec = None
        self._turn = 0

    def reset(self, task_id):
        prompt, spec = self._load(task_id)
        self._task_id, self._spec, self._turn = task_id, spec, 1
        return prompt

    def evaluate_once(self, task_id, kernel_c):
        return self._gate(task_id, kernel_c)

    def step(self, kernel_c):
        fb = self._gate(self._task_id, kernel_c)
        rew = R.reward_tier(fb, self._spec)
        obs = B.render(B.analyze(fb, self._spec, on_box=self.on_box), self._turn, self.max_turns)
        # done = reward tier 3 (correct AND genuine for every target mechanism), not merely
        # `correct`. A correct-but-not-genuine kernel (tier 2, e.g. correct but used_dma=false)
        # must keep looping so the mechanism-aware repair prescription (bottleneck.py's
        # "add dmstart/dmwait" etc.) actually reaches the model on the next turn. Ending on
        # bare correctness would short-circuit multi-turn repair before it ever fires.
        done = (rew >= 3) or self._turn >= self.max_turns
        self._turn += 1
        return obs, rew, done, fb
