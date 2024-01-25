import math
import random
import argparse

from femtogpt.loss import cross_entropy
from femtogpt.modules import GPT, MLP, Config, Embedding, LayerNorm, MultiHeadAttention
from femtogpt.utils import (
    add_matrices,
    concat,
    display_matrix,
    multiply_matrices,
    softmax,
    transpose,
)
from femtogpt.value import Value


def get_correct_sentence_proba(model, dataset, encode, softmax):
    total_proba = 0.0
    for i in range(len(dataset) - 1):
        prompt = dataset[: i + 1]
        target = dataset[i + 1]
        print(f"prompt: {prompt}; target: {target}")
        e = encode(prompt)
        logits = model(e)
        probs = softmax(logits[-1])
        probs = [p.data for p in probs]
        correct_log_prob = math.log(probs[encode(target)[0]])
        total_proba += correct_log_prob
        print(
            f"Correct token probability: {math.exp(correct_log_prob)} ; log_prob: {correct_log_prob}"
        )
        print("=" * 20)
    return total_proba


def train(dataset: str):
    print(f"Training on dataset: {dataset}")

    prompt = dataset[0]

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?! "
    vocab = set(alphabet)
    vocab_size = len(vocab)

    config = Config(vocab_size=vocab_size)

    stoi = {k: i for i, k in enumerate(vocab)}
    itos = {i: k for i, k in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda x: "".join([itos[t] for t in x])

    e = encode(dataset)
    x, y = e[:-1], e[1:]
    print(f"x={x}")
    print(f"y={y}")

    model = GPT(config=config)

    lr = 0.1
    losses = []
    total_steps = 100

    for k in range(total_steps):
        out = model(x)

        loss = cross_entropy(out, y)
        losses.append(loss.data)

        # decrease learning rate after 200 steps
        if k == 80:
            lr /= 2

        if k % 20 == 0:
            generation = model.generate(
                prompt, encode=encode, decode=decode, n=len(dataset) - 1
            )
            print(f"Step {k} / {total_steps} | Generation: {generation}")
        print(f"Step {k} / {total_steps} | Loss: {loss} | LR: {lr}")

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad

    try:
        from matplotlib import pyplot as plt

        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.show()
    except ImportError:
        print('Matplotlib is not installed. Run `pip install ".[viz]"`')

    print("=" * 20)
    print(
        f"The training is finished ✅. We now generate some text, starting with '{prompt}'."
    )
    generation = model.generate(
        prompt, encode=encode, decode=decode, n=len(dataset) - 1
    )
    print(f"Generated text: {generation}")
    print("=" * 20)

    correct_sentence_proba = get_correct_sentence_proba(model, dataset, encode, softmax)
    print(
        f"Probability of correct prediction given the first letter: {math.exp(correct_sentence_proba)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hello world")

    args = parser.parse_args()

    train(args.dataset)
