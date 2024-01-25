import torch as t
import torch.nn.functional as F

from femtogpt.loss import cross_entropy
from femtogpt.value import Value


# test with torch
def test_cross_entropy_loss():
    x = t.Tensor([[1, 2, 3], [4, 5, 6]]).double()
    y = t.Tensor([2, 1]).long()
    torch_loss = F.cross_entropy(x, y).data.item()

    print(torch_loss)

    x_2 = [[Value(1), Value(2), Value(3)], [Value(4), Value(5), Value(6)]]
    y_2 = [2, 1]
    my_loss = cross_entropy(x_2, y_2).data
    assert abs(torch_loss - my_loss) < 1e-6
