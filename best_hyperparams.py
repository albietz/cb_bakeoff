import argparse
import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
from collections import defaultdict
from eval_loss import load_names
from rank_algos import significance, significance_cs01, preprocess_df_granular, preprocess_df, base_name, set_base_name
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt


def find_winning_algo(df, algo_pattern, ds_ids, args=None):
    rgx = re.compile(algo_pattern)
    df = df.loc[df.algo.map(lambda s: rgx.match(s) is not None)]
    algos = list(np.sort(df.algo.unique()))
    print len(algos), 'algos'

    stats = {}
    for ds_id in ds_ids:
        a_l = df.loc[df.ds == ds_id, ['algo', 'sz', 'rawloss']].sort_values('algo')
        n = a_l.sz.max()
        loss = list(a_l.rawloss)
        wins = defaultdict(set)
        losses = defaultdict(set)

        for i in range(len(algos)):
            for j in range(i+1, len(algos)):
                pval = significance(loss[i], loss[j], n)
                if pval < args.alpha:
                    if loss[i] < loss[j]:
                        winner, loser = i, j
                    else:
                        winner, loser = j, i
                    wins[winner].add(loser)
                    losses[loser].add(winner)

        stats[ds_id] = (wins, losses)

    survivors = range(len(algos))
    ranked = []
    while survivors:
        scores = np.zeros(len(survivors))

        for ds in ds_ids:
            win_loss_diff = np.array([len(stats[ds][0][alg]) - len(stats[ds][1][alg])
                                      for alg in survivors])
            best = (win_loss_diff == win_loss_diff.max()).astype(np.int32)
            scores += best / best.sum()

        # print zip(np.array(algos)[np.array(survivors)], scores)
        # print 'losing algos:', np.array(algos)[np.array(survivors)[scores == scores.min()]]
        print np.sum(scores == scores.min()),
        # loser = survivors[scores.argmin()]
        loser_idx = np.random.choice(np.where(scores == scores.min())[0])
        loser = survivors[loser_idx]
        # print 'loser:', algos[loser]
        ranked.append(loser)
        survivors.remove(loser)
        for ds in ds_ids:
            for alg in survivors:
                if loser in stats[ds][0][alg]:
                    stats[ds][0][alg].remove(loser)
                if loser in stats[ds][1][alg]:
                    stats[ds][1][alg].remove(loser)

    print
    print [algos[i] for i in ranked[-3:]]
    print 'best:', algos[ranked[-1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find best fixed hyperparams')
    parser.add_argument('--sep_cb_type', action='store_true', default=False)
    parser.add_argument('--sep_name', action='store_true', default=False)
    parser.add_argument('--opt_b', action='store_true', default=False)
    parser.add_argument('--enc', default='neg10')
    parser.add_argument('--b', default=None)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--base_name', default='allrandfix')
    args = parser.parse_args()

    set_base_name(args.base_name)
    print(base_name())

    names = ['{}{}'.format(base_name(), name) for name in ['01', '01b', 'neg10', 'neg10b']]
    df = load_names(names, min_actions=None, min_size=None)

    df = preprocess_df_granular(df, all_algos=True, sep_enc=True, sep_b=not args.opt_b)

    enc_b_str = args.enc
    if not args.opt_b:
        if args.b is not None:
            enc_b_str += ':' + args.b
        else:
            enc_b_str += ':(b|nb)'

    ds_ids = df.ds.unique()
    np.random.seed(args.seed)
    np.random.shuffle(ds_ids)
    ds_ids = ds_ids[:200]
    # ds_ids = ds_ids[:262]
    np.save('ds_val_list.npy', ds_ids)
    np.random.seed(1337)

    df = df.loc[df.ds.map(lambda s: s in ds_ids)]

    print
    print 'greedy'
    pattern = 'epsilon:0:mtr:{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'regcb'
    pattern = 'regcb:c0:.*:mtr:{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'regcbopt'
    pattern = 'regcbopt:c0:.*:mtr:{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'cover_nu'
    pattern = 'cover:(4|8|16):psi:(0.01|0.1|1.0):nounif:(ips|dr):{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'cover'
    pattern = 'cover:(4|8|16):psi:(0.01|0.1|1.0):(ips|dr):{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'bag_greedy'
    pattern = 'bag:(4|8|16):greedify:(ips|dr|mtr):{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'bag'
    pattern = 'bag:(4|8|16):(ips|dr|mtr):{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'e_greedy'
    pattern = 'epsilon:(0.02|0.05|0.1):(ips|dr|mtr):{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)

    print
    print 'e_greedy_active'
    pattern = 'epsilon:[^:]*:nounifa:.*:(ips|dr|mtr):{}'.format(enc_b_str)
    find_winning_algo(df.copy(), pattern, ds_ids, args=args)
