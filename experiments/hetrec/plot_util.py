import json

import numpy as np

MARKERS = {
    'win': 's',
    'exp': '>',
    'adawin': 'D',
    'adaexp': '<',
    'avg': 'x',
    'true': 'o'
}

COLORS = {
    'win': 'C6',
    'exp': 'C1',
    'adawin': 'C5',
    'adaexp': 'C0',
    'avg': 'C3',
    'true': 'C2'
}

LABELS = {
    'win': '$\\textup{V}^{\\tau \\textup{IPS}}$',
    'exp': '$\\textup{V}^{\\alpha \\textup{IPS}}$',
    'adawin': 'Adaptive $\\textup{V}^{\\tau \\textup{IPS}}$',
    'adaexp': 'Adaptive $\\textup{V}^{\\alpha \\textup{IPS}}$',
    'avg': '$\\textup{V}^{\\textup{IPS}}$',
    'true': 'True'
}


def vstack_cut(arr):
    min_size = np.min([a.shape[0] for a in arr])
    return np.vstack([a[:min_size] for a in arr])


def get_policy_rewards(files, nr_policies=10):
    est = [{'x': [], 'y': []} for p in range(nr_policies)]
    true = [{'x': [], 'y': []} for p in range(nr_policies)]
    mse = [[] for p in range(nr_policies)]
    mae = [[] for p in range(nr_policies)]

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

    for p in range(nr_policies):
        est[p]['x'] = vstack_cut(est[p]['x'])[0, :]
        est[p]['ym'] = np.mean(vstack_cut(est[p]['y']), axis=0)
        est[p]['ys'] = np.std(vstack_cut(est[p]['y']), axis=0, ddof=1)
        true[p]['x'] = vstack_cut(true[p]['x'])[0, :]
        true[p]['ym'] = np.mean(vstack_cut(true[p]['y']), axis=0)
        true[p]['ys'] = np.std(vstack_cut(true[p]['y']), axis=0, ddof=1)

        mse[p] = {'mean': np.mean(mse[p]), 'std': np.std(mse[p], ddof=1), 'n': len(mse[p])}
        mae[p] = {'mean': np.mean(mae[p]), 'std': np.std(mae[p], ddof=1), 'n': len(mae[p])}

    return est, true, mse, mae



