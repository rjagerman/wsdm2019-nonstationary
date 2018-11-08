import json
from collections import defaultdict

import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},\\usepackage[libertine]{newtxmath},\\usepackage{sfmath},\\usepackage[T1]{fontenc},\\usepackage{siunitx}'
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


def vstack_cut(arr):
    min_size = np.min([a.shape[0] for a in arr])
    return np.vstack([a[:min_size] for a in arr])


def get_policy_rewards(files):
    est = [{'x': [], 'y': []} for p in range(10)]
    true = [{'x': [], 'y': []} for p in range(10)]
    mse = [[] for p in range(10)]
    mae = [[] for p in range(10)]

    for file in files:
        with open(file, 'rt') as f:
            data = json.load(f)

        for policy in range(10):
            p_x_est = np.array([d['iteration'] for d in data])
            p_est = np.array([d[f'policy_{policy}/estimate'] for d in data])
            p_x_true = np.array([d['iteration']
                            for d in data if f'policy_{policy}/true' in d])
            p_true = np.array([d[f'policy_{policy}/true']
                            for d in data if f'policy_{policy}/true' in d])

            est[policy]['x'].append(p_x_est)
            est[policy]['y'].append(p_est)

            true[policy]['x'].append(p_x_true)
            true[policy]['y'].append(p_true)

            # p_true_interpolated = np.interp(p_x_est, p_x_true, p_true)
            # average_mse.append(np.mean((p_true_interpolated - p_est) ** 2))

            p_est_mse = p_est[np.in1d(p_x_est, p_x_true)]
            p_true_int = np.interp(p_x_est, p_x_true, p_true)

            mse[policy].append(np.mean((p_true_int - p_est) ** 2))
            mae[policy].append(np.mean(np.abs(p_true_int - p_est)))

    for p in range(10):
        est[p]['x'] = vstack_cut(est[p]['x'])[0, :]
        est[p]['ym'] = np.mean(vstack_cut(est[p]['y']), axis=0)
        est[p]['ys'] = np.std(vstack_cut(est[p]['y']), axis=0, ddof=1)
        true[p]['x'] = vstack_cut(true[p]['x'])[0, :]
        true[p]['ym'] = np.mean(vstack_cut(true[p]['y']), axis=0)
        true[p]['ys'] = np.std(vstack_cut(true[p]['y']), axis=0, ddof=1)

        mse[p] = {'mean': np.mean(mse[p]), 'std': np.std(mse[p], ddof=1), 'n': len(mse[p])}
        mae[p] = {'mean': np.mean(mae[p]), 'std': np.std(mae[p], ddof=1), 'n': len(mae[p])}

    return est, true, mse, mae


def main():
    parser = ArgumentParser()
    parser.add_argument("--file1", nargs='+', type=str, default=None)
    parser.add_argument("--file2", nargs='+', type=str, default=None)
    parser.add_argument("--file3", nargs='+', type=str, default=None)
    parser.add_argument("--file4", nargs='+', type=str, default=None)
    parser.add_argument("--true", nargs='+', type=str, default=None)
    parser.add_argument("--policy", type=int, default=6)
    parser.add_argument("--name1", type=str, default="")
    parser.add_argument("--name2", type=str, default="")
    parser.add_argument("--name3", type=str, default="")
    parser.add_argument("--name4", type=str, default="")
    parser.add_argument("--ylim", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="plot.pdf")
    parser.add_argument("--legend", action='store_true')
    args = parser.parse_args()

    markevery = 1000
    markersize = 4
    linewidth = 1.1
    p = args.policy

    if args.file1 is not None:
        est, true, mse, mae = get_policy_rewards(args.file1)
        plt.plot(est[p]['x'], est[p]['ym'], color='C0', marker='>',
                 label=args.name1, markevery=(333, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C0')
        print(f"{args.name1}: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.file2 is not None:
        est, true, mse, mae = get_policy_rewards(args.file2)
        plt.plot(est[p]['x'], est[p]['ym'], color='C1', marker='v',
                 label=args.name2, markevery=markevery,
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C1')
        print(f"{args.name2}: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.file3 is not None:
        est, true, mse, mae = get_policy_rewards(args.file3)
        plt.plot(est[p]['x'], est[p]['ym'], color='C3', marker='<',
                 label=args.name3, markevery=(666, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C3')
        print(f"{args.name3}: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.file4 is not None:
        est, true, mse, mae = get_policy_rewards(args.file4)
        plt.plot(est[p]['x'], est[p]['ym'], color='C0', marker='^',
                 label=args.name4, markevery=markevery,
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C0')
        print(f"{args.name4}: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.true is not None:
        est, true, mse, mae = get_policy_rewards(args.true)
        plt.plot(true[p]['x'], true[p]['ym'], color='C2', marker='o',
                 label='True', markersize=markersize, markevery=int(markevery / 100), linewidth=linewidth)
        plt.plot()
        plt.fill_between(true[p]['x'], true[p]['ym'] - true[p]['ys'],
                         true[p]['ym'] + true[p]['ys'], alpha=0.35, color='C2')

    xmargin = 25000
    plt.ylim((0, args.ylim))
    plt.xlim((0 - xmargin, 1000000 + xmargin))
    plt.ylabel("Reward")
    plt.xlabel("Time $t$")
    artists = []
    if args.legend:
        lgd = plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.48), framealpha=1.0)
        artists.append(lgd)
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"\\num[group-separator={{,}}]{{{int(x)}}}"))
    ax.grid(axis='x', color='k', linestyle='dotted', linewidth=1.0, alpha=0.35)

    fig = plt.gcf()
    fig.set_size_inches(4.0, 1.25)
    # fig.tight_layout()
    fig.savefig(args.output, dpi=30, bbox_inches='tight',
                additional_artists=artists)


if __name__ == "__main__":
    main()
