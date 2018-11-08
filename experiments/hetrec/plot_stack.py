from argparse import ArgumentParser
from glob import glob
from experiments.hetrec.plot_util import get_policy_rewards, MARKERS, COLORS,\
    LABELS
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{libertine},\\usepackage[libertine]{newtxmath},\\usepackage{sfmath},\\usepackage[T1]{fontenc},\\usepackage{siunitx}'
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def main():
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--policy", type=int, default=6)
    parser.add_argument("--settings", nargs='+', type=str, default=['stationary', 'abrupt', 'linear'])
    parser.add_argument("--methods", nargs='+', type=str, default=['exp', 'avg',
                                                                   'adaexp'])
    parser.add_argument("--ylim", type=float, default=0.4)
    parser.add_argument("--legend", action='store_true')
    parser.add_argument("--output", type=str, default="plot.pdf")
    args = parser.parse_args()

    markevery = 1000
    markersize = 4
    linewidth = 1.1
    p = args.policy

    shared_ax = None
    for i, setting in enumerate(args.settings):
        ax = plt.subplot(313 - i, sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        true = None

        for method in args.methods:
            files = glob(f"{args.folder}/{setting}/{method}/*/log.json")
            est, true, mse, mae = get_policy_rewards(files)
            plt.plot(est[p]['x'], est[p]['ym'], color=COLORS[method],
                     marker=MARKERS[method], label=LABELS[method],
                     markevery=(333, markevery), markersize=markersize,
                     linewidth=linewidth)
            plt.fill_between(est[p]['x'], est[p]['ym'] - est[p]['ys'],
                             est[p]['ym'] + est[p]['ys'], alpha=0.35,
                             color=COLORS[method])

        if true is not None:
            plt.plot(true[p]['x'], true[p]['ym'], color=COLORS['true'],
                     marker=MARKERS['true'], label=LABELS['true'],
                     markevery=int(markevery / 100), markersize=markersize,
                     linewidth=linewidth)
            plt.fill_between(true[p]['x'], true[p]['ym'] - true[p]['ys'],
                             true[p]['ym'] + true[p]['ys'], alpha=0.35,
                             color=COLORS['true'])

        if i > 0:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_xticklines(), visible=False)
        else:
            plt.xlabel("Time $t$")
        ax.grid(axis='x', color='k', linestyle='dotted', linewidth=1.0, alpha=0.35)
        plt.ylim((0, args.ylim))
        plt.ylabel("Reward")

    xmargin = 25000
    plt.xlim((0 - xmargin, 1000000 + xmargin))

    artists = []
    if args.legend:
        lines = []
        for method in ['true'] + args.methods:
            line = mlines.Line2D([], [], color=COLORS[method], marker=MARKERS[method],
                                 linewidth=linewidth, markersize=markersize,
                                 label=LABELS[method])
            lines.append(line)
        lgd = plt.legend(handles = lines, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -2.43),
                         handleheight=1.05)
        artists.append(lgd)
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(
        lambda x, loc: f"\\num[group-separator={{,}}]{{{int(x)}}}"))
    fig = plt.gcf()
    fig.set_size_inches(4.0, 3 * 1.34)
    fig.savefig(args.output, dpi=30, bbox_inches='tight',
                additional_artists=artists)


if __name__ == "__main__":
    main()
