import torch as t
import torch.nn as nn

from femtogpt.modules import LayerNorm


def test_layer_norm():
    x = t.randn(2, 6)
    ln = nn.LayerNorm(6, elementwise_affine=False, bias=False)
    x_norm = ln(x)

    ln2 = LayerNorm(6)
    x_norm2 = ln2(x)

    for i in range(x_norm.shape[0]):
        for j in range(x_norm.shape[1]):
            assert abs(x_norm[i, j] - x_norm2[i][j].data) < 1e-4
