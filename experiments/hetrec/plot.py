import json
from collections import defaultdict

import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},\\usepackage[libertine]{newtxmath},\\usepackage{sfmath},\\usepackage[T1]{fontenc},\\usepackage{siunitx}'
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


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
    parser.add_argument("--adaexp", nargs='+', type=str, default=None)
    parser.add_argument("--adawin", nargs='+', type=str, default=None)
    parser.add_argument("--win", nargs='+', type=str, default=None)
    parser.add_argument("--exp", nargs='+', type=str, default=None)
    parser.add_argument("--avg", nargs='+', type=str, default=None)
    parser.add_argument("--true", nargs='+', type=str, default=None)
    parser.add_argument("--policy", type=int, default=6)
    parser.add_argument("--ylim", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="plot.pdf")
    parser.add_argument("--legend", action='store_true')
    args = parser.parse_args()

    markevery = 1000
    markersize = 4
    linewidth = 1.1
    p = args.policy

    if args.win is not None:
        est, true, mse, mae = get_policy_rewards(args.win)
        plt.plot(est[p]['x'], est[p]['ym'], color='C6', marker='s',
                 label='$\\textup{V}^{\\tau \\textup{IPS}}$', markevery=(333, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C6')
        print(f"win MSE: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.exp is not None:
        est, true, mse, mae = get_policy_rewards(args.exp)
        plt.plot(est[p]['x'], est[p]['ym'], color='C1', marker='>',
                 label='$\\textup{V}^{\\alpha \\textup{IPS}}$', markevery=(333, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C1')
        print(f"exp MSE: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.adawin is not None:
        est, true, mse, mae = get_policy_rewards(args.adawin)
        plt.plot(est[p]['x'], est[p]['ym'], color='C5', marker='D',
                 label='Adaptive $\\textup{V}^{\\tau \\textup{IPS}}$', markevery=(666, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C5')
        print(f"adawin MSE: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.adaexp is not None:
        est, true, mse, mae = get_policy_rewards(args.adaexp)
        plt.plot(est[p]['x'], est[p]['ym'], color='C0', marker='<',
                 label='Adaptive $\\textup{V}^{\\alpha \\textup{IPS}}$', markevery=(666, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C0')
        print(f"adaexp MSE: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.avg is not None:
        est, true, mse, mae = get_policy_rewards(args.avg)
        plt.plot(est[p]['x'], est[p]['ym'], color='C3', marker='x',
                 label='$\\textup{V}^{\\textup{IPS}}$', markevery=(500, markevery),
                 markersize=markersize, linewidth=linewidth)
        plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                         est[p]['ym'] + est[p]['ys'], alpha=0.35, color='C3')
        print(f"avg MSE: {mse[p]['mean']} (std: {mse[p]['std']}, n: {mse[p]['n']})")

    if args.true is not None:
        est, true, mse, mae = get_policy_rewards(args.true)
        plt.plot(true[p]['x'], true[p]['ym'], color='C2', marker='o',
                 label='True', markersize=markersize, linewidth=linewidth,
                 markevery=int(markevery / 100))
        plt.plot()
        plt.fill_between(true[p]['x'], true[p]['ym'] - true[p]['ys'],
                         true[p]['ym'] + true[p]['ys'], alpha=0.35, color='C2')

    xmargin = 25000
    plt.ylim((0, args.ylim))
    plt.xlim((0 - xmargin, 1000000 + xmargin))
    plt.ylabel("Reward $\\in [0, 1]$")
    plt.xlabel("Time $t$")
    artists = []
    if args.legend:
        lwin = mlines.Line2D([], [], color='C6', marker='s', linewidth=linewidth, markersize=markersize, label='$\\textup{V}^{\\tau \\textup{IPS}}$')
        lexp = mlines.Line2D([], [], color='C1', marker='>', linewidth=linewidth, markersize=markersize, label='$\\textup{V}^{\\alpha \\textup{IPS}}$')
        ladawin = mlines.Line2D([], [], color='C5', marker='D', linewidth=linewidth, markersize=markersize, label='Adaptive $\\textup{V}^{\\tau \\textup{IPS}}$')
        ladaexp = mlines.Line2D([], [], color='C0', marker='<', linewidth=linewidth, markersize=markersize, label='Adaptive $\\textup{V}^{\\alpha \\textup{IPS}}$')
        lavg = mlines.Line2D([], [], color='C3', marker='x', linewidth=linewidth, markersize=markersize, label='$\\textup{V}^{\\textup{IPS}}$')
        ltrue = mlines.Line2D([], [], color='C2', marker='o', linewidth=linewidth, markersize=markersize, label='True')
        lines = [ltrue, lavg, lwin, lexp, ladawin, ladaexp]
        lgd = plt.legend(handles=lines, loc='lower center', ncol=3,
                         bbox_to_anchor=(0.5, 1.05), handleheight=1.05)
        artists.append(lgd)
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"\\num[group-separator={{,}}]{{{int(x)}}}"))
    fig = plt.gcf()
    fig.set_size_inches(4.0, 1.4)
    # fig.tight_layout()
    fig.savefig(args.output, dpi=30, bbox_inches='tight',
                additional_artists=artists)


if __name__ == "__main__":
    main()
