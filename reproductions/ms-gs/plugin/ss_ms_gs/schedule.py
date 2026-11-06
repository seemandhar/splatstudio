"""Multi-scale training schedule (C4), official MS-GS train.py:82-194 with --ms_train --insert_large.

The official loop draws a whole "epoch" (a shuffled stack of every training camera) at one
resolution. When the stack is empty it picks the next resolution index: 0 (1x) with probability
0.75, otherwise uniformly among the *least-trained* indices of the enabled list
{0} + [[2, 3], [4, 5], [6, 7]][j] for every insertion step inc_at[j] < iteration. Index 1 (2x)
is never trained with --insert_large. Non-1x iterations are weighted by 0.1 in the loss.

The native runtime samples views uniformly with replacement; this sampler reproduces the
resolution sequence (epoch length = number of training views) on top of it.
"""

import random

# train.py:98-111 (inc_reso_idx_train, inc_reso_idx, inc_reso_at = base_iter + 10/20/30).
INSERT_TABLES = {
    5: ([[2, 3], [4], [5]], [2, 4], [1010, 1020]),
    6: ([[2, 3], [4, 5], [6]], [2, 4, 6], [1010, 1020, 1030]),
    7: ([[2, 3], [4, 5], [6, 7]], [2, 4, 6], [1010, 1020, 1030]),
}


def insert_table(max_scale):
    if max_scale not in INSERT_TABLES:
        raise ValueError("max_scale must be 5, 6 or 7 (official train.py)")
    return INSERT_TABLES[max_scale]


class MultiScaleSampler:
    def __init__(self, max_scale=7, p_full_res=0.75, seed=0):
        self.train_lists, self.insert_levels, self.insert_at = insert_table(max_scale)
        self.n_levels = max_scale + 1
        self.p_full_res = p_full_res
        self.rng = random.Random(seed)
        self.n_train = 0
        self.left = 0
        self.level = 0
        self.iterations = [0] * self.n_levels

    def enabled(self, iteration):
        out = [0]
        # train.py:171 `iteration > inc_reso_at` (zip truncates like the official index mask)
        for at, lst in zip(self.insert_at, self.train_lists):
            if iteration > at:
                out += lst
        return out

    def next_level(self, iteration):
        """Resolution index for training iteration `iteration` (1-based)."""
        if self.n_train <= 0:
            return 0
        if self.left == 0:
            if self.rng.random() < self.p_full_res:
                self.level = 0
            else:
                cand = self.enabled(iteration)
                least = min(self.iterations[i] for i in cand)
                self.level = self.rng.choice([i for i in cand if self.iterations[i] == least])
            self.left = self.n_train
        self.left -= 1
        self.iterations[self.level] += 1
        return self.level

    def state_dict(self):
        return {"left": self.left, "level": self.level, "iterations": list(self.iterations)}


def position_lr(initial, step, max_steps=30000, final_ratio=0.01):
    """3DGS get_expon_lr_func (lr_delay_steps 0): log-linear to final over max_steps, then hold."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return initial * final_ratio**t
