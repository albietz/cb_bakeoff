import eval_common

import argparse
import os
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd

USE_ADF = True
USE_CS = False

DIR_PATTERN_CS = '/scratch/clear/abietti/cb_eval/res_cs/cbresults_{}/'
DIR_PATTERN = '/scratch/clear/abietti/cb_eval/res/cbresults_{}/'
# DIR_PATTERN = '/bscratch/b-albiet/cbresults_{}/'


# functions for interactive queries
def print_best_algos_for_ds(df, ds, n=50):
    print df.loc[df.ds == ds, ['na', 'sz', 'nf', 'algo', 'lr', 'loss', 'rawloss']].sort_values('loss').head(n).to_string()


def print_win_ds_for_algo(df, algo_pattern):
    '''e.g. print_win_ds_for_algo(dfbest, 'bag:.*:mtr') shows datasets where
    bag + mtr with tuned learning rate wins
    '''
    rgx = re.compile(algo_pattern)
    print df.loc[df.groupby('ds').loss.idxmin()].loc[
            df.algo.map(lambda x: rgx.match(x) is not None),
            ['ds', 'na', 'sz', 'nf', 'algo', 'lr', 'loss']].to_string()


def load_names(names, cb_type=None, normalize=True, min_actions=None, min_size=None, ty='best', use_cs=False):
    '''
    ty: best, all or raw
    '''
    cache_file = os.path.join('/scratch/clear/abietti/cb_eval/rescache{}'.format('_cs' if use_cs else ''),
            '{}:{}:{}:{}:{}:{}.pkl'.format('_'.join(names), cb_type, normalize,
                      min_actions, min_size, ty))
    if os.path.exists(cache_file): # return cached
        return pd.read_pickle(cache_file)

    def cached(df):
        df.to_pickle(cache_file)
        return df

    df_raws = []
    dir_pattern = DIR_PATTERN_CS if use_cs else DIR_PATTERN
    for name in names:
        resdir = dir_pattern.format(name)
        loss_file = os.path.join(resdir, 'all_losses.txt.gz')
        df_raw = eval_common.load_raw(
                loss_file, adf=USE_ADF, cb_type=cb_type,
                min_actions=min_actions, min_size=min_size)
        df_raw['name'] = name
        df_raws.append(df_raw)
    df_raw = pd.concat(df_raws, ignore_index=True)
    df_raw['rawloss'] = df_raw['loss']

    if normalize:
        ref = df_raw[df_raw.algo == 'supervised'].groupby('ds').loss.min()
        for k, loss in ref.iteritems():
            df_raw.loc[df_raw.ds == k, ['loss']] = \
                    (df_raw.loc[df_raw.ds == k, ['loss']] - loss) / loss

    if ty == 'raw':
        return cached(df_raw)

    # remove supervised results
    df = df_raw[df_raw.algo != 'supervised'].copy()
    if ty == 'all':
        return cached(df)

    # best learning rate per dataset/algo/name
    dfbest = df.loc[df.groupby(['ds', 'algo', 'name']).loss.idxmin()].copy().sample(frac=1)
    return cached(dfbest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval losses')
    # parser.add_argument('--results_dir', default='/scratch/clear/abietti/cb_eval/res/cbresults_covmtr/')
    parser.add_argument('--results_dir', default='/bscratch/b-albiet/cbresults_loss910b_adc/')
    parser.add_argument('--name', default=None)
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--cb_type', default=None)
    parser.add_argument('--interactive', action='store_true',
            help='break execution after loading dataframes (for interactive analysis)')
    parser.add_argument('--min_actions', type=int, default=None)
    parser.add_argument('--min_size', type=int, default=None)
    args = parser.parse_args()

    if args.name is not None:
        args.results_dir = DIR_PATTERN.format(args.name)

    loss_file = os.path.join(args.results_dir, 'all_losses.txt')
    if os.path.exists(loss_file + '.gz'): # prefer gzipped file if it's there
        loss_file += '.gz'
    if not os.path.exists(loss_file):
        sys.stderr.write('concatenating loss files...')
        import subprocess
        subprocess.check_call(['cat {} | sort > {}'.format(os.path.join(args.results_dir, 'loss*.txt'),
                               loss_file)], shell=True)

    df_raw = eval_common.load_raw(loss_file, adf=USE_ADF, cb_type=args.cb_type,
                                  min_actions=args.min_actions, min_size=args.min_size)
    df_raw['rawloss'] = df_raw['loss']

    if not args.median:
        if args.weighted:
            def print_results(df, dfbest):
                print '***** average loss per algo after tuning lr for each (ds, algo)'
                by = dfbest.algo
                print ((dfbest.loss * dfbest.weights).groupby(by).sum() / dfbest.weights.groupby(by).sum()).sort_values()
                print
                print '***** average loss per (algo, lr)'
                by = [df.algo, df.lr]
                print ((df.loss * df.weights).groupby(by).sum() / df.weights.groupby(by).sum()).sort_values()
                print
                print '***** average_ds of best loss across all (algo, lr):',
                # print df.groupby('ds').loss.min().mean()
        else:
            def print_results(df, dfbest):
                print '***** average loss per algo after tuning lr for each (ds, algo)'
                print dfbest.groupby('algo').loss.mean().sort_values()
                print
                print '***** average loss per (algo, lr)'
                print df.groupby(['algo', 'lr']).loss.mean().sort_values()
                print
                print '***** average_ds of best loss across all (algo, lr):',
                print df.groupby('ds').loss.min().mean()
    else:
        def print_results(df, dfbest):
            print '***** median loss per algo after tuning lr for each (ds, algo)'
            print dfbest.groupby('algo').loss.median().sort_values()
            print
            print '***** median loss per (algo, lr)'
            print df.groupby(['algo', 'lr']).loss.median().sort_values()
            print
            print '***** median_ds of best loss across all (algo, lr):',
            print df.groupby('ds').loss.min().median()

    if args.normalize:
        pass # df_raw.loss *= df_raw.sz.apply(np.sqrt)
    if args.weighted:
        df_raw['weights'] = df_raw.sz.apply(np.sqrt)

    # remove supervised results
    df = df_raw[df_raw.algo != 'supervised'].copy()

    # best learning rate per dataset/algo
    dfbest = df.loc[df.groupby(['ds', 'algo']).loss.idxmin()].copy().sample(frac=1)

    # if args.interactive:
    #     sys.exit(0)

    if not args.interactive:
        # print '######################## wins by (algo, lr)'
        # print df.loc[df.groupby('ds').loss.idxmin()].groupby(['algo', 'lr']).size().sort_values(ascending=False)
        # print
        print '######################## wins by algo after tuning lr'
        print dfbest.loc[dfbest.groupby('ds').loss.idxmin()].groupby('algo').size().sort_values(ascending=False)

        print
        print
        # print '######################## Progressive Validation loss #############################'
        # print_results(df, dfbest)

        print
        print
        print '######################## PV loss minus supervised', '(normalized)' if args.normalize else ''
        # ref = df_raw[df_raw.algo == 'supervised'].groupby(['ds', 'lr']).loss.min()
        # for k, loss in ref.iteritems():
        #     idxs = (df.ds == k[0]) & (df.lr == k[1])
        #     df.loc[idxs, ['loss']] -= loss

    ref = df_raw[df_raw.algo == 'supervised'].groupby('ds').loss.min()
    refmax = df_raw.groupby('ds').loss.max()
    for k, loss in ref.iteritems():
        df.loc[df.ds == k, ['loss']] -= loss
        if args.normalize:
            df.loc[df.ds == k, ['loss']] /= ( loss)
            # df.loc[df.ds == k, ['loss']] /= (refmax[k] - loss)
        idxs = dfbest.ds == k
        dfbest.loc[idxs, ['loss']] -= loss
        if args.normalize:
            dfbest.loc[idxs, ['loss']] /= ( loss)

    if not args.interactive:
        print_results(df, dfbest)
    sys.exit(0)

    print
    print
    print '######################## PV loss minus eps-greedy 0.05'
    # reset df
    df = df_raw[df_raw.algo != 'supervised'].copy()
    dfbest = df.loc[df.groupby(['ds', 'algo']).loss.idxmin()].copy()

    # ref = df[df.algo == 'epsilon:0.05'].groupby(['ds', 'lr']).loss.min()
    # for k, loss in ref.iteritems():
    #     idxs = (df.ds == k[0]) & (df.lr == k[1])
    #     df.loc[idxs, ['loss']] -= loss

    ref = dfbest[dfbest.algo == 'epsilon:0.05'].groupby('ds').loss.min()
    for k, loss in ref.iteritems():
        df.loc[df.ds == k, ['loss']] -= loss
        idxs = dfbest.ds == k
        dfbest.loc[idxs, ['loss']] -= loss

    print_results(df, dfbest)

    print
    print
    print '######################## PV loss minus best'
    # reset df
    df = df_raw[df_raw.algo != 'supervised'].copy()
    dfbest = df.loc[df.groupby(['ds', 'algo']).loss.idxmin()].copy()

    # ref = df.groupby(['ds', 'lr']).loss.min()
    # for k, loss in ref.iteritems():
    #     idxs = (df.ds == k[0]) & (df.lr == k[1])
    #     df.loc[idxs, ['loss']] -= loss

    ref = dfbest.groupby('ds').loss.min()
    for k, loss in ref.iteritems():
        df.loc[df.ds == k, ['loss']] -= loss
        idxs = dfbest.ds == k
        dfbest.loc[idxs, ['loss']] -= loss

    print_results(df, dfbest)
