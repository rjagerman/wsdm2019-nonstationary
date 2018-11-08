import json
import numpy as np
from argparse import ArgumentParser


def mse(data):
    mse = []
    for d in data:
        for p in range(10):
            if f'policy_{p}/true' in d:
                true = d[f'policy_{p}/true']
                est = d[f'policy_{p}/estimate']
                mse.append((true - est) ** 2)
    return np.mean(mse)


def main():
    parser = ArgumentParser()
    parser.add_argument("--files", nargs="+", type=str, required=True)
    args = parser.parse_args()

    best_score = 1.0
    best_file = ''
    for file in args.files:
        with open(file, 'rt') as f:
            data = json.load(f)
            score = mse(data)
            if score < best_score:
                best_score = score
                best_file = file
            print(f"{file}: {score}")
    print(f"best: {best_file}: {best_score}")


if __name__ == "__main__":
    main()
