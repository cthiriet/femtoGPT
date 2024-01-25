from femtogpt.utils import add_matrices, concat, multiply_matrices, softmax, transpose
from femtogpt.value import Value


def test_softmax():
    logits = [Value(1), Value(2), Value(3)]
    probs = softmax(logits)
    assert len(probs) == len(logits)
    assert abs(sum([p.data for p in probs]) - 1) < 1e-6


def test_multiply_matrices():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = multiply_matrices(A, B)
    assert C == [[19, 22], [43, 50]]


def test_add_matrices():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = add_matrices(A, B)
    assert C == [[6, 8], [10, 12]]


def test_transpose():
    A = [[1, 2], [3, 4]]
    At = transpose(A)
    assert At == [[1, 3], [2, 4]]


def test_concat():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = concat(A, B)
    assert C == [[1, 2, 5, 6], [3, 4, 7, 8]]
