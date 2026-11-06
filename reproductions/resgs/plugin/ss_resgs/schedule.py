"""Stage / substage state machine of the official ResGS training loop.

Port of the counters in `train.py:training` (github.com/yanzhelyu/ResGS): `warm_up_iter`,
`cur_level`, `change_split_level_iter`, `last_change_iter`, `next_change_iter` and
`scene.cur_resolution`. The per-iteration call order is kept:

    render at stage `res_index`                        (renderer, before the step)
    stats_active(t) / densify_step(t)                  (densification block)
    change_stage(t)                                    (`if iteration in change_iter`)
    opacity_reduce_step(t)                             (AbsGS periodic opacity reduction)
    end_iteration(t)                                   (warm-up and substage counters)
"""


class ResGSSchedule:
    def __init__(self, c):
        self.change_iter = [int(v) for v in c["change_iter"]]
        self.resolution_scales = [int(v) for v in c["resolution_scales"]]
        self.stage_split = int(c["stage_split"])
        self.warm_up = int(c["warm_up_iter"])
        self.start_stat = int(c["start_stat"])
        self.update_from = int(c["update_from"])
        self.update_interval = int(c["update_interval"])
        self.update_until = int(c["update_until"])
        self.reduce_interval = int(c["opacity_reduce_interval"])
        self.prune_until = int(c["prune_until"])
        self.use_opacity_reduce = bool(c["use_opacity_reduce"])
        if len(self.change_iter) != len(self.resolution_scales) - 1:
            raise ValueError("change_iter needs one entry per stage transition")
        if self.stage_split < 1:
            raise ValueError("stage_split must be >= 1")
        self.res_index = 0  # scene.cur_resolution
        self.warm = 0  # warm_up_iter
        self.cur_level = 1  # substage k
        last, nxt = self.update_from, self.change_iter[0]
        self.split_counter = self._split(last, nxt)  # change_split_level_iter
        self.last_change, self.next_change = last, nxt

    def _split(self, last, nxt):
        return (nxt - last) // self.stage_split  # get_change_split_iter

    # ---- queries ------------------------------------------------------------------------
    @property
    def pyramid_level(self):
        """Number of pyrDown applications for the current stage (resolution_scales entry)."""
        return self.resolution_scales[self.res_index]

    def stats_active(self, t):
        return self.start_stat < t < self.update_until and self.warm == 0

    def densify_step(self, t):
        return self.stats_active(t) and t > self.update_from and t % self.update_interval == 0

    def opacity_reduce_step(self, t):
        """(reduce, prune_first) for iteration t; call after change_stage(t)."""
        if self.res_index != 0 and self.use_opacity_reduce and t < self.prune_until:
            if t % self.reduce_interval == 0:
                return True, t >= self.update_until
        return False, False

    # ---- transitions --------------------------------------------------------------------
    def change_stage(self, t):
        if t not in self.change_iter:
            return False
        max_level = len(self.resolution_scales)
        if self.res_index + 1 < max_level:  # scene.up_one_resolution
            self.res_index += 1
        self.warm = self.warm_up
        if self.res_index != max_level - 1:
            self.next_change = self.change_iter[self.res_index]
        else:
            self.next_change = self.update_until
        self.last_change = self.change_iter[self.res_index - 1] + self.warm
        self.cur_level = self.stage_split * self.res_index + 1
        self.split_counter = self._split(self.last_change, self.next_change)
        return True

    def end_iteration(self, t):
        if self.warm != 0:
            self.warm -= 1
        if (self.split_counter != 0 and self.warm == 0
                and self.update_from < t < self.update_until):
            self.split_counter -= 1
            if self.split_counter == 0:
                self.cur_level += 1
                self.split_counter = self._split(self.last_change, self.next_change)

    def to_dict(self):
        return {"res_index": self.res_index, "warm": self.warm, "cur_level": self.cur_level,
                "split_counter": self.split_counter, "last_change": self.last_change,
                "next_change": self.next_change}
