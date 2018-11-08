import json
from collections import defaultdict

import numpy as np
from scipy import stats
from argparse import ArgumentParser
from glob import glob


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

            mse[policy].append(np.mean((p_true - p_est_mse) ** 2))
            mae[policy].append(np.mean(np.abs(p_true - p_est_mse)))

    mse_f = [(np.mean([mse[p][f] for p in range(10)])) for f in range(len(files))]
    mae_f = [(np.mean([mae[p][f] for p in range(10)])) for f in range(len(files))]
    #mse_f = {'mean': np.mean(mse_f), 'std': np.std(mse_f, ddof=1), 'n': len(files)}
    #mae_f = {'mean': np.mean(mae_f), 'std': np.std(mae_f, ddof=1), 'n': len(files)}

    for p in range(10):
        est[p]['x'] = vstack_cut(est[p]['x'])[0, :]
        est[p]['ym'] = np.mean(vstack_cut(est[p]['y']), axis=0)
        est[p]['ys'] = np.std(vstack_cut(est[p]['y']), axis=0, ddof=1)
        true[p]['x'] = vstack_cut(true[p]['x'])[0, :]
        true[p]['ym'] = np.mean(vstack_cut(true[p]['y']), axis=0)
        true[p]['ys'] = np.std(vstack_cut(true[p]['y']), axis=0, ddof=1)

        mse[p] = {'mean': np.mean(mse[p]), 'std': np.std(mse[p], ddof=1), 'n': len(mse[p])}
        mae[p] = {'mean': np.mean(mae[p]), 'std': np.std(mae[p], ddof=1), 'n': len(mae[p])}

    return est, true, mse, mae, mse_f, mae_f


def statistical_significance_test(values, values_to_compare):

    # This is a 2-tailed paired t-test for checking mean differences:
    result = stats.ttest_rel(values, values_to_compare)
    pvalue = result.pvalue

    # Test direction of improvement:
    delta = np.array(values) - np.array(values_to_compare)
    direction = np.mean(delta) > 0.0
    return direction, pvalue


def main():
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    args = parser.parse_args()

    mult = 1000.0

    settings = ['linear', 'abrupt', 'stationary']
    methods = ['avg', 'win', 'adawin', 'exp', 'adaexp']

    setting_names = {
        'linear': 'Smooth',
        'abrupt': 'Abrupt',
        'stationary': 'Stationary'
    }
    method_names = {
        'avg': '$\\Vips$',
        'win': '$\\Vwindow$',
        'adawin': '$\\Vwindow$ (adaptive)',
        'exp': '$\\Vexp$',
        'adaexp': '$\\Vexp$ (adaptive)'
    }

    all = {k: {k2: None for k2 in settings} for k in methods}

    print(f"\\toprule")
    print(f"Estimator & {' & '.join(setting_names.values())} \\\\")
    print(f"\\midrule")
    for method in methods:
        print(method_names[method], end=' ')
        for setting in settings:
            files = glob(f"{args.folder}/{setting}/{method}/*/log.json")
            files = sorted(files)
            est, true, mse, mae, mse_f, mae_f = get_policy_rewards(files)
            all[method][setting] = mse_f

            print(f"& {(mult * np.mean(mse_f)):.3f}", end=' ')

            significance_markers = ""
            if method != "avg" and all['avg'][setting] is not None:
                direction, pvalue = statistical_significance_test(mse_f, all['avg'][setting])
                if pvalue < 0.01:
                    significance_markers += " \\triangle"
                    if not direction:
                        significance_markers += "down"

            if method.startswith("ada") and all[method[3:]][setting] is not None:
                direction, pvalue = statistical_significance_test(mse_f, all[method[3:]][setting])
                if pvalue < 0.01:
                    significance_markers += " \\blacktriangle"
                    if not direction:
                        significance_markers += "down"

            if len(significance_markers) > 0:
                print(f"$^{{{significance_markers}}}$", end=' ')
        print("\\\\")
    print("\\bottomrule")



if __name__ == "__main__":
    main()
