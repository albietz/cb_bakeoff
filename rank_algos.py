import argparse
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
from collections import defaultdict
from eval_loss import load_names
from scipy.special import erf, erfinv

_base_name = 'disagree'

def base_name():
    global _base_name
    return _base_name

def set_base_name(name):
    global _base_name
    _base_name = name
    print('base name set to', _base_name)

def significance(x, y, sz):
    diff = x - y
    se = 1e-6 + np.sqrt((x * (1 - x) + y * (1 - y))/ sz)
    pval = 1 - erf(np.abs(diff / se))
    return pval

def significance_cs01(x, y, sz):
    '''p-value for losses in [0,1] (Hoeffding).'''
    diff = x - y
    return 2. * np.exp(- sz * (diff ** 2) / 8)

algo_list_old = [
        'epsilon:0:dr',
        'epsilon:0:mtr',
        'bag:16:dr',
        'bag:16:mtr',
        'bag:16:greedify:dr',
        'bag:16:greedify:mtr',
        'bag:8:dr',
        'bag:8:mtr',
        'bag:8:greedify:dr',
        'bag:8:greedify:mtr',
        'bag:4:mtr',
        'bag:4:greedify:mtr',
        # 'bag:8:dr',
        # 'bag:4:mtr',
        'cover:4:psi:0.01:nounif:dr',
        'cover:8:psi:0.01:nounif:dr',
        'cover:16:psi:0.01:nounif:dr',
        'cover:16:psi:0.01:dr',
        # 'cover:4:psi:1.0:nounif:dr',
        # 'cover:16:psi:1.0:nounif:dr',
        'epsilon:0.05:nounifa:c0:1e-06:dr',
        'epsilon:0.05:nounifa:c0:1e-06:ips',
        'epsilon:1:nounifa:c0:1e-06:dr',
        'epsilon:1:nounifa:c0:1e-06:ips',
        ]
psi = '0.1'
algo_list = ['epsilon:0:mtr',
             'epsilon:0:dr',
             'cover:16:psi:{}:nounif:dr'.format(psi),
             'bag:16:greedify:mtr',
             'epsilon:0.02:mtr',
             'cover:16:psi:{}:dr'.format(psi),
             'epsilon:1:nounifa:c0:1e-06:dr']

def preprocess_df_granular(df, sep_name=False, sep_enc=False, sep_b=False, sep_lr=False,
                           all_algos=False, algos=None):
    if not all_algos:
        if algos is None:
            algos = algo_list
        df = df.loc[df.algo.map(lambda x: x in algos)]

    if sep_name:
        df.algo = df.algo + ':' + df.name.map(lambda s: s.replace(base_name(), ''))
    if sep_enc:
        df.algo = df.algo + ':' + df.name.map(lambda s: s.replace(base_name(), '').replace('b', ''))
    if sep_b:
        df.algo = df.algo + ':' + df.name.map(lambda s: 'b' if s.endswith('b') else 'nb')
    if sep_lr:
        df.algo = df.algo + ':lr:' + df.lr.map(lambda s: s.replace('learning_rate:', ''))

    # aggregate best
    df = df.loc[df.groupby(['ds', 'algo']).rawloss.idxmin()].copy()

    # df.algo = df.algo.map(lambda s: s.replace('epsilon', 'aepsilon'))

    return df


def preprocess_df(df, sep_reduction=False, sep_name=False, sep_enc=False, sep_b=False,
                  reduce_algo=False, filter_algos=None):
    rgxs = [
        ('epsilon:0:ips', 'remove'),   # remove ips for greedy
        ('epsilon:0:dr', 'remove'),   # remove ips for greedy
        ('epsilon:0:mtr', 'greedy'),
        ('bag:2:.*', 'remove'),
        ('bag:.*:greedify:.*', 'bag_greedy'),
        ('bag:.*', 'bag'),
        ('epsilon:.*nounifa.*:mtr', 'remove'),  # remove MTR for greedy active
        ('epsilon:.*nounifa.*', 'e_greedy_active'),  # 'remove'
        ('epsilon:.*', 'e_greedy'),
        ('cover:1:.*nounif.*', 'remove'),
        ('cover:.*:psi:0:.*', 'remove'),      # cover_greedy, remove b/c weird
        ('cover:1:.*', 'e_greedy_decay'),
        ('cover:.*:mtr', 'cover_nounif'),    # remove mtr for cover
        ('cover:.*:nounif:.*', 'cover_nounif'),
        ('cover:.*', 'cover'),
        ('regcb:.*', 'regcb'),
        ('regcbopt:.*', 'regcbopt'),
    ]

    # sep_reduction = False  # separate names for each reduction
    # sep_name = True # separate names for each "name" (0/1, etc)

    for pattern, algname in rgxs:
        # if sep_reduction and algname != 'remove':
        #     for red in ['mtr', 'dr', 'ips']:
        #         rgx = re.compile(pattern + red)
        #         df.loc[df.algo.map(lambda x: rgx.match(x) is not None), ['algo']] = algname + '_' + red
        # else:
        rgx = re.compile(pattern)
        df.loc[df.algo.map(lambda x: rgx.match(x) is not None), ['algo']] = algname

    df = df.loc[df.algo != 'remove']

    if filter_algos:
        df = df.loc[df.algo.map(lambda x: x in filter_algos)]

    if reduce_algo:
        df.algo = ''

    if sep_reduction:
        df.algo = df.algo + ':' + df.cb_type

    if sep_name:
        df.algo = df.algo + ':' + df.name.map(lambda s: s.replace(base_name(), ''))
    if sep_enc:
        df.algo = df.algo + ':' + df.name.map(lambda s: s.replace(base_name(), '').replace('b', ''))
    if sep_b:
        df.algo = df.algo + ':' + df.name.map(lambda s: 'b' if s.endswith('b') else 'nb')

    # take best performing within each group (reduce by min)
    df = df.loc[df.groupby(['ds', 'algo']).rawloss.idxmin()].copy()
    return df

def scatterplot(df, alg_names, labels=None, raw=False, alpha=0.05, lim_min=0., lim_max=1.):
    assert len(alg_names) == 2
    if labels is None:
        labels = alg_names

    rawx = df.loc[df.algo == alg_names[0]].groupby('ds').rawloss.mean()
    rawy = df.loc[df.algo == alg_names[1]].groupby('ds').rawloss.mean()
    if raw:
        x, y = rawx, rawy
    else:
        x = df.loc[df.algo == alg_names[0]].groupby('ds').loss.mean()
        y = df.loc[df.algo == alg_names[1]].groupby('ds').loss.mean()
    sz = df.loc[df.algo == alg_names[0]].groupby('ds').sz.max()
    pvals = significance(rawx, rawy, sz)

    import matplotlib.pyplot as plt
    plt.scatter(x, y, s=plt.rcParams['lines.markersize']**2 * (pvals < alpha).map(lambda x: 1.0 if x else 0.2),
                c=(pvals < alpha).map(lambda x: 'r' if x else 'k'))
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max], [lim_min, lim_max],color='k')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rank algorithms')
    parser.add_argument('names', default=None, help='names, comma-separated')
    parser.add_argument('--reload', action='store_true', default=False)
    parser.add_argument('--reload_data', default='stats_tmp.pkl')
    parser.add_argument('--sep_cb_type', action='store_true', default=False)
    parser.add_argument('--sep_name', action='store_true', default=False)
    parser.add_argument('--granular', action='store_true', default=False)
    parser.add_argument('--sep_lr', action='store_true', default=False)
    parser.add_argument('--all_algos', action='store_true', default=False)
    parser.add_argument('--keep_datasets', action='store_true', default=False,
                        help='keeps comparing on all datasets at each iteration')
    parser.add_argument('--reduce_algo', action='store_true', default=False)
    parser.add_argument('--filter_algos', default=None)
    parser.add_argument('--interactive', action='store_true', default=False)
    parser.add_argument('--interactive_norank', action='store_true', default=False)
    args = parser.parse_args()


    if not args.reload:
        df = load_names(args.names.split(','), min_actions=None,
                        ty='all' if args.sep_lr else 'best')
        # df = df.loc[df.algo.map(lambda x: x.startswith('epsilon:') or x.startswith('bag:'))]
        # df = df.loc[df.algo.map(lambda x: x in algo_list)].copy()
        # df = df.loc[df.na * df.nf > 200]
        if args.granular:
            df = preprocess_df_granular(df, sep_name=args.sep_name,
                                        sep_lr=args.sep_lr, all_algos=args.all_algos)
        else:
            df = preprocess_df(df, sep_reduction=args.sep_cb_type, sep_name=args.sep_name,
                               reduce_algo=args.reduce_algo, filter_algos=args.filter_algos.split(',') if args.filter_algos else None)

        ds_ids = df.ds.unique()
        ds_to_sz = pickle.load(open('ds_sz.pkl'))

        if args.interactive_norank:
            sys.exit(0)

        stats = {}
        algos = None
        for ds_id in ds_ids:
            n = ds_to_sz[ds_id]
            a_l = df.loc[df.ds == ds_id, ['algo', 'rawloss']].sort_values('algo')
            if algos is None:
                algos = list(a_l.algo)
            loss = list(a_l.rawloss)
            wins = defaultdict(set)
            losses = defaultdict(set)

            for i in range(len(algos)):
                for j in range(i+1, len(algos)):
                    pval = significance(loss[i], loss[j], n)
                    if pval < 0.05:
                        if loss[i] < loss[j]:
                            winner, loser = i, j
                        else:
                            winner, loser = j, i
                        wins[winner].add(loser)
                        losses[loser].add(winner)

            stats[ds_id] = (wins, losses)

        pickle.dump((stats, algos, ds_ids), open(args.reload_data, 'w'))
    else:
        (stats, algos, ds_ids) = pickle.load(open(args.reload_data))

    if args.interactive:
        sys.exit(0)

    best_algos = []
    # surviving datasets and algorithms
    surv_ds = np.array(ds_ids)
    surv_algs = list(range(len(algos)))
    while len(surv_algs) > 0 and len(surv_ds) > 0:
        X = np.array([[len(stats[ds_id][0][j]) for j in surv_algs] for ds_id in surv_ds])
        max_per_ds = X.max(1)
        alg_counts = (X == X.max(1)[:,None]).sum(0)
        print('stats', {algos[surv_algs[i]]: c for i, c in enumerate(alg_counts)})
        best_idx = alg_counts.argmax()
        best = surv_algs[best_idx]
        print('overall best:', algos[best], ',', end=' ')
        best_algos.append((best, alg_counts.max()))

        # remove covered datasets
        if not args.keep_datasets:
            surv_ds = surv_ds[X[:,best_idx] != max_per_ds]
        surv_algs.remove(best)
        print('remaining datasets', len(surv_ds))
        for ds_id in surv_ds:
            for j in surv_algs:
                if best in stats[ds_id][0][j]:
                    stats[ds_id][0][j].remove(best)

    for i, c in best_algos:
        print(algos[i], '({})'.format(c), '->', end=' ')
