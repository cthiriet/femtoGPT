from femtogpt.utils import softmax
from femtogpt.value import Value


def cross_entropy(logits, targets):
    n_samples = len(logits)
    cross_entropy_sum = Value(0.0)

    for l, t in zip(logits, targets):
        y_pred = softmax(l)
        cross_entropy_sum += y_pred[t].log()

    # Average cross entropy over all samples
    return -cross_entropy_sum / n_samples
