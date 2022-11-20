def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr