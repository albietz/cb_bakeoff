import argparse
from eval_loss import load_names
from rank_algos import significance, preprocess_df_granular, preprocess_df
import numpy as np

MTR_LABEL = 'iwr'

def wins_losses(df, xname, yname, alpha=0.01):
    rawx = df.loc[df.algo == xname].groupby('ds').rawloss.mean()
    rawy = df.loc[df.algo == yname].groupby('ds').rawloss.mean()
    sz = df.loc[df.algo == xname].groupby('ds').sz.max()
    pvals = significance(rawx, rawy, sz)

    return (np.sum((rawx < rawy) & (pvals < alpha)),
            np.sum((rawx > rawy) & (pvals < alpha)))


def print_table(df, alg_names, labels=None, args=None):
    n = len(alg_names)
    if labels is None:
        labels = alg_names
    table = [['-' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            wins, losses = wins_losses(df, alg_names[i], alg_names[j], alpha=args.alpha)
            if args.diff:
                table[i][j] = str(wins - losses)
                table[j][i] = str(losses - wins)
            else:
                table[i][j] = '{} / {}'.format(wins, losses)
                table[j][i] = '{} / {}'.format(losses, wins)

    print r'\begin{tabular}{ | l |', 'c | ' * n, '}'
    print r'\hline'
    print r'$\downarrow$ vs $\rightarrow$ &', ' & '.join(labels), r'\\ \hline'
    for i in range(n):
        print labels[i], '&', ' & '.join(table[i]), r'\\ \hline'
    print r'\end{tabular}'


def print_table_rect(df, alg_names_row, alg_names_col, labels_row=None, labels_col=None, args=None):
    n, m = len(alg_names_row), len(alg_names_col)
    if labels_row is None:
        labels_row = alg_names_row
    if labels_col is None:
        labels_col = alg_names_col
    table = [['-' for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            wins, losses = wins_losses(df, alg_names_row[i], alg_names_col[j], alpha=args.alpha)
            if args.diff:
                table[i][j] = str(wins - losses)
            else:
                table[i][j] = '{} / {}'.format(wins, losses)

    print r'\begin{tabular}{ | l |', 'c | ' * m, '}'
    print r'\hline'
    print r'$\downarrow$ vs $\rightarrow$ &', ' & '.join(labels_col), r'\\ \hline'
    for i in range(n):
        print labels_row[i], '&', ' & '.join(table[i]), r'\\ \hline'
    print r'\end{tabular}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='barplots')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--granular_opt', action='store_true', default=False)
    parser.add_argument('--granular', action='store_true', default=False)
    parser.add_argument('--granular_neg10', action='store_true', default=False)
    parser.add_argument('--granular_01', action='store_true', default=False)
    parser.add_argument('--granular_neg10_bopt', action='store_true', default=False)
    parser.add_argument('--granular_name', action='store_true', default=False)
    parser.add_argument('--bag_vs_greedy', action='store_true', default=False)
    parser.add_argument('--opt', action='store_true', default=False)
    parser.add_argument('--opt_neg10', action='store_true', default=False)
    parser.add_argument('--opt_01', action='store_true', default=False)
    parser.add_argument('--opt_algo', action='store_true', default=False)
    parser.add_argument('--sep_cb_type', action='store_true', default=False)
    parser.add_argument('--sep_name', action='store_true', default=False)
    parser.add_argument('--sep_enc', action='store_true', default=False)
    parser.add_argument('--sep_b', action='store_true', default=False)
    parser.add_argument('--algo', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--enc', default=None)
    parser.add_argument('--b', default=None)
    parser.add_argument('--cb_type', default=None)
    parser.add_argument('--alpha', default=0.05)
    parser.add_argument('--min_size', type=int, default=None)
    parser.add_argument('--diff', action='store_true', default=True)
    parser.add_argument('--nodiff', dest='diff', action='store_false')
    args = parser.parse_args()

    names = ['disagree01', 'disagree01b', 'disagreeneg10', 'disagreeneg10b']
    df = load_names(names, min_actions=None, min_size=args.min_size)

    psi = '0.1'
    if args.granular_opt or args.all:
        df_all = preprocess_df_granular(df, all_algos=True)

        # optimized name
        print 'optimized over encoding/baseline'
        algs = ['epsilon:0:mtr', 'epsilon:0:dr', 'cover:16:psi:{}:nounif:dr'.format(psi),
                'bag:16:mtr', 'bag:16:greedify:mtr', 'epsilon:0.02:mtr', 'cover:16:psi:{}:dr'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular or args.all:
        print
        print 'best fixed encoding/baseline'
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs = ['epsilon:0:mtr:neg10b', 'epsilon:0:dr:neg10b', 'cover:16:psi:{}:nounif:dr:neg10'.format(psi),
                'bag:16:mtr:01b', 'bag:16:greedify:mtr:01b', 'epsilon:0.02:mtr:neg10',
                'cover:16:psi:{}:dr:neg10'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr:01']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular_neg10 or args.all:
        print
        print 'fixed -1/0, fixed baseline choice (01 for active)'
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs = ['epsilon:0:mtr:neg10b', 'epsilon:0:dr:neg10b', 'cover:16:psi:{}:nounif:dr:neg10'.format(psi),
                'bag:16:mtr:neg10b', 'bag:16:greedify:mtr:neg10b', 'epsilon:0.02:mtr:neg10',
                'cover:16:psi:{}:dr:neg10'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr:01']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular_01 or args.all:
        print
        print 'fixed 0/1, fixed baseline choice'
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs = ['epsilon:0:mtr:01b', 'epsilon:0:dr:01b', 'cover:16:psi:{}:nounif:dr:01'.format(psi),
                'bag:16:mtr:01b', 'bag:16:greedify:mtr:01b', 'epsilon:0.02:mtr:01',
                'cover:16:psi:{}:dr:01'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr:01']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular_neg10_bopt or args.all:
        print
        print 'fixed -1/0, baseline optimized (01 for active)'
        df_all = preprocess_df_granular(df, all_algos=True, sep_enc=True)

        algs = ['epsilon:0:mtr:neg10', 'epsilon:0:dr:neg10', 'cover:16:psi:{}:nounif:dr:neg10'.format(psi),
                'bag:16:mtr:neg10', 'bag:16:greedify:mtr:neg10', 'epsilon:0.02:mtr:neg10',
                'cover:16:psi:{}:dr:neg10'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr:01']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular_name or args.all:
        print
        print 'fixed name', args.name
        assert args.name is not None, 'must specify --name'
        name = 'disagree' + args.name
        df_all = df.loc[df.name == name]
        df_all = preprocess_df_granular(df_all, all_algos=True)

        algs = ['epsilon:0:mtr', 'epsilon:0:dr', 'cover:16:psi:{}:nounif:dr'.format(psi),
                'bag:16:mtr', 'bag:16:greedify:mtr', 'epsilon:0.02:mtr',
                'cover:16:psi:{}:dr'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.bag_vs_greedy or args.all:
        print
        print 'bag/bag-g vs greedy'
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs_row = ['epsilon:0:mtr:neg10b', 'epsilon:0:dr:neg10b']
        labels_row = ['G-{}'.format(MTR_LABEL), 'G-dr']
        bag_algs = ['bag:{}:mtr:01b', 'bag:{}:greedify:mtr:01b',
                    'bag:{}:mtr:neg10b', 'bag:{}:greedify:mtr:neg10b']
        bag_labels = ['{}', '{}-g']
        print '0/1 + b'
        algs_col = [x.format(s) for s in ['4', '8', '16'] for x in bag_algs[:2]]
        labels_col = [x.format(s) for s in ['4', '8', '16'] for x in bag_labels]
        print_table_rect(df_all, algs_row, algs_col, labels_row, labels_col, args=args)
        print_table(df_all, algs_col[:2], labels_col[:2], args=args)
        print_table(df_all, algs_col[2:4], labels_col[2:4], args=args)
        print_table(df_all, algs_col[4:], labels_col[4:], args=args)

        print '-1/0 + b'
        algs_col = [x.format(s) for s in ['4', '8', '16'] for x in bag_algs[2:]]
        print_table_rect(df_all, algs_row, algs_col, labels_row, labels_col, args=args)
        print_table(df_all, algs_col[:2], labels_col[:2], args=args)
        print_table(df_all, algs_col[2:4], labels_col[2:4], args=args)
        print_table(df_all, algs_col[4:], labels_col[4:], args=args)

    if args.opt or args.all:
        print
        print 'optimize hyperparams, encoding/baseline'
        df_all = preprocess_df(df)

        algs = ['greedy', 'cover_nounif',
                'bag', 'bag_greedy', 'e_greedy',
                'cover', 'e_greedy_active']
        labels = ['G', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels=labels, args=args)

    if args.opt_neg10 or args.all:
        print
        print 'optimize hyperparams, encoding/baseline'
        df_all = preprocess_df(df, sep_enc=True)

        algs = ['greedy', 'cover_nounif',
                'bag', 'bag_greedy', 'e_greedy',
                'cover', 'e_greedy_active:01']
        for i in range(len(algs) - 1):
            algs[i] += ':neg10'
        labels = ['G', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels=labels, args=args)

    if args.opt_01 or args.all:
        print
        print 'optimize hyperparams, encoding/baseline'
        df_all = preprocess_df(df, sep_enc=True)

        algs = ['greedy', 'cover_nounif',
                'bag', 'bag_greedy', 'e_greedy',
                'cover', 'e_greedy_active']
        for i in range(len(algs)):
            algs[i] += ':01'
        labels = ['G', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels=labels, args=args)

    if args.opt_algo or args.all:
        print
        print 'optimize hyperparams, fixed encoding/baseline'
        df_all = preprocess_df(df, sep_name=args.sep_name or args.name,
                               sep_b=args.sep_b or args.b,
                               sep_enc=args.sep_enc or args.enc,
                               sep_reduction=args.sep_cb_type or args.cb_type)

        algo = args.algo
        algs = [algo]
        labels = None
        if args.sep_cb_type or args.cb_type:
            cb_types = [args.cb_type] if args.cb_type else ['ips', 'dr', 'mtr']
            algs = [a + ':' + red for a in algs for red in cb_types]
            if args.sep_cb_type:
                labels = [s.replace('mtr', MTR_LABEL) for s in cb_types]
        if args.sep_name or args.name:
            names = [args.name] if args.name else ['01', '01b', 'neg10', 'neg10b']
            algs = [a + ':' + name for a in algs for name in names]
            if args.sep_name:
                labels = [s.replace('neg', '-') for s in names]
        if args.sep_enc or args.enc:
            encs = [args.enc] if args.enc else ['01', 'neg10']
            algs = [a + ':' + enc for a in algs for enc in encs]
            if args.sep_enc:
                labels = [s.replace('neg', '-') for s in encs]
        if args.sep_b or args.b:
            bs = [args.b] if args.b else ['b', 'nb']
            algs = [a + ':' + b for a in algs for b in bs]
            if args.sep_b:
                labels = bs
        print_table(df_all, algs, labels=labels, args=args)
