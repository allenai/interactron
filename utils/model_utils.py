def merge_batch_seq(x):
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])


def unmerge_batch_seq(x, b, s):
    return x.view(b, s, *x.shape[1:])
