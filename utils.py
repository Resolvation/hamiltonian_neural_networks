def linear_lr(epoch, n_epochs):
    if epoch <= n_epochs / 2:
        return 1
    else:
        k = (n_epochs - epoch) / (n_epochs / 2)
        return k + 0.1 * (1 - k)


def change_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr