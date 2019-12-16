import math

from torch.optim.lr_scheduler import _LRScheduler


class TruncateCosineScheduler(_LRScheduler):

    def __init__(self, optimizer,
                 n_steps, n_cycles,
                 annealing=True,
                 last_epoch=-1):
        self.n_steps = n_steps
        self.n_cycles = n_cycles
        self.annealing = annealing
        self.last_epoch = last_epoch
        super(TruncateCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        param = 1
        epochs_per_cycle = math.floor(self.n_steps / self.n_cycles)

        if self.annealing:
            param = 1 + self.last_epoch / self.n_steps
            epochs_per_cycle *= 1 + param

        cos_inner = math.pi * (self.last_epoch % epochs_per_cycle) / epochs_per_cycle

        return [base_lr / (2 * param) * (math.cos(cos_inner) + 1)
                for base_lr in self.base_lrs]


def cosine_lr_func_gen(n_steps, n_cycles,
                       lrate_max, lr_min=1e-7,
                       annealing=True):
    """Generates a learning rate function of truncated cosine type,
       if annealing is True (default) as epochs progress
       amplitude decreases and cycle length increases, otherwise cycle remains the same"""

    def cosine_learning_rate(epoch):
        """Function generated with the parameters retrieved from cosine_lr_func_gen"""

        param = 1
        epochs_per_cycle = math.floor(n_steps / n_cycles)

        if annealing:
            param = 1 + epoch / n_steps
            epochs_per_cycle *= 1 + param

        cos_inner = math.pi * ((2 * epoch) % epochs_per_cycle) / epochs_per_cycle

        new_lr = lrate_max/(2 * param) * (math.cos(cos_inner) + 1)

        return new_lr + lr_min

    return cosine_learning_rate


def adjust_learning_rate(optimizer, new_lr):
    """
    Changes learning rate in optimizer

    :param optimizer: pytorch optimizer
    :param new_lr: scalar value
    """
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = new_lr


if __name__ == '__main__':

    from argparse import ArgumentParser
    import matplotlib.pyplot as plt

    PARSER = ArgumentParser()
    PARSER.add_argument('--lr_max', default=0.001, type=float)
    PARSER.add_argument('--cycles', default=15, type=int)
    PARSER.add_argument('--steps', default=1000, type=int)
    ARGS = PARSER.parse_args()

    LR_FUNC1 = cosine_lr_func_gen(10000, 4, 0.01, 1e-6)
    LR_FUNC2 = cosine_lr_func_gen(10, 5, 0.5, 1e-5, annealing=False)

    LR_EVOLUTION1 = [LR_FUNC1(i) for i in range(11448)]
    LR_EVOLUTION2 = [LR_FUNC2(i) for i in range(100)]

    print(min(LR_EVOLUTION1))

    plt.plot([i for i in range(11448)], LR_EVOLUTION1)
    # plt.plot([i for i in range(100)], LR_EVOLUTION2)
    plt.show()
