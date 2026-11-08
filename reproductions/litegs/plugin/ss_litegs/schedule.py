"""LiteGS epoch-based schedules (official litegs/training/trainer.py, densify.py).

The official trainer iterates in epochs (one shuffled pass over the training views). The native
runtime counts 1-based steps, so step s belongs to epoch (s-1)//n_train and epoch e ends at step
(e+1)*n_train. `SHARED` carries n_train and the last completed step from the density controller
(which receives the training cameras) to the renderer (which applies the SH schedule).
"""

SHARED = {"n_train": None, "step": 0}


def bind(n_train):
    if int(n_train) < 1:
        raise ValueError("n_train must be >= 1")
    SHARED["n_train"] = int(n_train)
    SHARED["step"] = 0


def reset():
    SHARED["n_train"] = None
    SHARED["step"] = 0


def epoch_of(step, n_train):
    return (int(step) - 1) // int(n_train)


def total_epochs(total_steps, n_train):
    """trainer.py: total_epoch = int(op.iterations / len(trainingset))."""
    return int(total_steps / n_train)


def densify_until(total_epoch, decay_interval, until_fraction=0.8):
    """trainer.py: int(total_epoch*0.8/opacity_reset_interval)*opacity_reset_interval+1."""
    return int(total_epoch * until_fraction / decay_interval) * decay_interval + 1


def in_window(epoch, start, until):
    return start <= epoch < until


def is_densify_epoch(epoch, start, until, interval):
    """densify.py is_densify_actived (also gates statistics collection, StatisticsHelper)."""
    return in_window(epoch, start, until) and epoch % interval == 0


def is_decay_epoch(epoch, start, until, interval):
    return in_window(epoch, start, until) and epoch % interval == 0


def sh_degree(step, n_train, interval=5, max_degree=3):
    """trainer.py: actived_sh_degree = min(int(epoch/5), sh_degree) at the start of each epoch."""
    return min(epoch_of(step, n_train) // interval, max_degree)
