import math
import random
from dataclasses import dataclass

from femtogpt.utils import (
    add_matrices,
    concat,
    display_matrix,
    multiply_matrices,
    shape,
    softmax,
    transpose,
)
from femtogpt.value import Value


@dataclass
class Config:
    vocab_size: int
    n_embed: int = 8
    n_head: int = 2
    n_layer: int = 1
    context_len: int = 12
    head_dim: int = n_embed // n_head


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, nin, bias=True, nonlin=False):
        self.scale = math.sqrt(1 / nin)
        self.w = [Value(random.uniform(-1, 1) * self.scale) for _ in range(nin)]
        self.b = Value(0) if bias else None
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b or 0)
        return act.relu() if self.nonlin else act

    def parameters(self):
        if self.b is None:
            return self.w
        else:
            return self.w + [self.b]


class Linear(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class Embedding(Module):
    def __init__(self, vocab_size, n_embed):
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.scale = math.sqrt(1 / n_embed)
        self.w = [
            [Value(random.uniform(-1, 1) * self.scale) for _ in range(n_embed)]
            for _ in range(vocab_size)
        ]

    def __call__(self, x):
        return [self.w[i] for i in x]

    def parameters(self):
        return [p for row in self.w for p in row]


class Head(Module):
    def __init__(self, n_embed, head_dim):
        self.n_embed = n_embed
        self.head_dim = head_dim
        self.key = Linear(n_embed, head_dim, bias=False)
        self.query = Linear(n_embed, head_dim, bias=False)
        self.value = Linear(n_embed, head_dim, bias=False)

    def __call__(self, x):
        # for each token in the input sequence
        # compute the key, query, and value
        K = [self.key(t) for t in x]
        Q = [self.query(t) for t in x]
        V = [self.value(t) for t in x]

        # compute the similarity matrix
        sim = multiply_matrices(Q, transpose(K))

        for i in range(len(sim)):
            for j in range(len(sim[i])):
                # scale (softmax is sensitive to outliers)
                sim[i][j] /= Value(math.sqrt(self.head_dim))

                # mask (ignore the future tokens)
                if j > i:
                    sim[i][j] = Value(-1e9)

        # apply softmax
        sim = [softmax(row) for row in sim]

        res = multiply_matrices(sim, V)

        return res

    def parameters(self):
        return self.key.parameters() + self.query.parameters() + self.value.parameters()


class MultiHeadAttention(Module):
    def __init__(self, n_embed, n_head):
        self.n_embed = n_embed
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.heads = [Head(n_embed, self.head_dim) for _ in range(n_head)]
        self.linear = Linear(n_embed, n_embed)

    def __call__(self, x):
        # apply each attention head
        heads = [att(x) for att in self.heads]

        # check the shape of the output
        for h in heads:
            assert shape(h)[0] == len(x)  # context length
            assert shape(h)[1] == self.head_dim  # head dimension

        # concatenate the results
        res = heads[0]
        for i in range(1, len(heads)):
            res = concat(res, heads[i])

        # project back to the original dimensionality
        res = [self.linear(t) for t in res]

        return res

    def parameters(self):
        return self.linear.parameters() + [
            p for h in self.heads for p in h.parameters()
        ]


class MLP(Module):
    def __init__(self, n_embed, ff_factor=4):
        self.linear1 = Linear(n_embed, n_embed * ff_factor, nonlin=True)
        self.linear2 = Linear(n_embed * ff_factor, n_embed)

    def __call__(self, x):
        x = [self.linear1(t) for t in x]
        x = [self.linear2(t) for t in x]
        return x

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()


class LayerNorm(Module):
    def __init__(self, nin):
        self.nin = nin
        self.gamma = Value(1)
        self.beta = Value(0)

    def __call__(self, x):
        normalized_x = []
        for t in x:
            mean = sum(t) / self.nin
            var = sum((ti - mean) ** 2 for ti in t) / self.nin
            norm_example = [(ti - mean) / ((var + 1e-9) ** 0.5) for ti in t]
            scaled_shifted_example = [
                (self.gamma * ti) + self.beta for ti in norm_example
            ]
            normalized_x.append(scaled_shifted_example)
        return normalized_x

    def parameters(self):
        return [self.gamma, self.beta]


class Block(Module):
    def __init__(self, n_embed, n_head, ff_factor=4):
        self.mha = MultiHeadAttention(n_embed, n_head)
        self.mlp = MLP(n_embed, ff_factor)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)

    def __call__(self, x):
        x = add_matrices(x, self.mha(self.ln1(x)))
        x = add_matrices(x, self.mlp(self.ln2(x)))
        return x

    def parameters(self):
        return (
            self.mha.parameters()
            + self.mlp.parameters()
            + self.ln1.parameters()
            + self.ln2.parameters()
        )


class GPT(Module):
    def __init__(self, config: Config):
        self.config = config
        self.emb = Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = Embedding(config.context_len, config.n_embed)
        self.blocks = [
            Block(config.n_embed, config.n_head) for _ in range(config.n_layer)
        ]
        self.ln_f = LayerNorm(config.n_embed)
        self.unembed = Linear(config.n_embed, config.vocab_size)

    def __call__(self, x):
        # embedding layer
        E = self.emb(x)

        # positional encoding layer
        P = self.pos_emb(list(range(len(x))))

        # residual stream
        residual_stream = add_matrices(E, P)

        for block in self.blocks:
            residual_stream = block(residual_stream)

        # final layer norm
        residual_stream = self.ln_f(residual_stream)

        # linear layer
        logits = [self.unembed(t) for t in residual_stream]

        return logits

    def generate(self, prompt, encode, decode, n=10):
        x = encode(prompt)
        for _ in range(n):
            out = self(x)
            probs = softmax(out[-1])
            probs = [p.data for p in probs]
            next_token = random.choices(range(self.config.vocab_size), weights=probs)[0]
            x.append(next_token)
        return decode(x)

    def parameters(self):
        return (
            self.emb.parameters()
            + self.pos_emb.parameters()
            + [p for block in self.blocks for p in block.parameters()]
            + self.ln_f.parameters()
            + self.unembed.parameters()
        )
