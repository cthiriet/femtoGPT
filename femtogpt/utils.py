def shape(x):
    return len(x), len(x[0])


def display_matrix(x):
    for row in x:
        print([f"{r.data:.2f}" for r in row])
    print(f"shape: {shape(x)}")


def softmax(logits):
    counts = [logit.exp() for logit in logits]
    s = sum(counts)
    return [c / s for c in counts]


def concat(A, B):
    A_h, A_w = shape(A)
    B_h, B_w = shape(B)
    assert A_h == B_h
    return [A[i] + B[i] for i in range(A_h)]


def transpose(A):
    A_h, A_w = shape(A)
    return [[A[i][j] for i in range(A_h)] for j in range(A_w)]


def add_matrices(A, B):
    A_h, A_w = shape(A)
    B_h, B_w = shape(B)
    assert A_h == B_h and A_w == B_w
    C = [[0.0 for _ in range(A_w)] for _ in range(A_h)]
    for i in range(A_h):
        for j in range(A_w):
            C[i][j] = A[i][j] + B[i][j]
    return C


def multiply_matrices(A, B):
    A_h, A_w = shape(A)
    B_h, B_w = shape(B)
    assert A_w == B_h
    C = [[0.0 for _ in range(B_w)] for _ in range(A_h)]
    for i in range(A_h):
        for j in range(B_w):
            for k in range(A_w):
                C[i][j] += A[i][k] * B[k][j]
    return C
