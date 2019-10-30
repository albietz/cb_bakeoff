import argparse
from eval_loss import load_names
from rank_algos import significance, significance_cs01, preprocess_df_granular, preprocess_df, base_name, set_base_name
import numpy as np

MTR_LABEL = 'iwr'

def wins_losses(df, xname, yname, args=None):
    rawx = df.loc[df.algo == xname].groupby('ds').rawloss.mean()
    rawy = df.loc[df.algo == yname].groupby('ds').rawloss.mean()
    sz = df.loc[df.algo == xname].groupby('ds').sz.max()
    if args.use_hoeffding:
        pvals = significance_cs01(rawx, rawy, sz)
    else:
        pvals = significance(rawx, rawy, sz)

    return (np.sum((rawx < rawy) & (pvals < args.alpha)),
            np.sum((rawx > rawy) & (pvals < args.alpha)))


def print_table(df, alg_names, labels=None, args=None):
    n = len(alg_names)
    if labels is None:
        labels = alg_names
    table = [['-' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            wins, losses = wins_losses(df, alg_names[i], alg_names[j], args=args)
            if args.diff:
                table[i][j] = str(wins - losses)
                table[j][i] = str(losses - wins)
            else:
                table[i][j] = '{} / {}'.format(wins, losses)
                table[j][i] = '{} / {}'.format(losses, wins)

    print(r'\begin{tabular}{ | l |', 'c | ' * n, '}')
    print(r'\hline')
    print(r'$\downarrow$ vs $\rightarrow$ &', ' & '.join(labels), r'\\ \hline')
    for i in range(n):
        print(labels[i], '&', ' & '.join(table[i]), r'\\ \hline')
    print(r'\end{tabular}')


def print_table_rect(df, alg_names_row, alg_names_col, labels_row=None, labels_col=None, args=None):
    n, m = len(alg_names_row), len(alg_names_col)
    if labels_row is None:
        labels_row = alg_names_row
    if labels_col is None:
        labels_col = alg_names_col
    table = [['-' for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            wins, losses = wins_losses(df, alg_names_row[i], alg_names_col[j], args=args)
            if args.diff:
                table[i][j] = str(wins - losses)
            else:
                table[i][j] = '{} / {}'.format(wins, losses)

    print(r'\begin{tabular}{ | l |', 'c | ' * m, '}')
    print(r'\hline')
    print(r'$\downarrow$ vs $\rightarrow$ &', ' & '.join(labels_col), r'\\ \hline')
    for i in range(n):
        print(labels_row[i], '&', ' & '.join(table[i]), r'\\ \hline')
    print(r'\end{tabular}')


def print_enc_table(df, df_big, algs, labels=None, args=None):
    n = len(algs)
    if labels is None:
        labels = alg_names
    table = [['-' for _ in range(n)] for _ in range(2)]

    for i in range(n):
        wins, losses = wins_losses(df, algs[i].format(enc='neg10'), algs[i].format(enc='01'), args=args)
        if args.diff:
            table[0][i] = str(wins - losses)
        else:
            table[0][i] = '{} / {}'.format(wins, losses)
        wins, losses = wins_losses(df_big, algs[i].format(enc='neg10'), algs[i].format(enc='01'), args=args)
        if args.diff:
            table[1][i] = str(wins - losses)
        else:
            table[1][i] = '{} / {}'.format(wins, losses)

    print(r'\begin{tabular}{ | c |', 'c | ' * n, '}')
    print(r'\hline')
    print(r'datasets &', ' & '.join(labels), r'\\ \hline')
    print('all', '&', ' & '.join(table[0]), r'\\ \hline')
    print(r'$\geq$ 10000', '&', ' & '.join(table[1]), r'\\ \hline')
    print(r'\end{tabular}')


def print_loss_table(df, algs, labels=None, args=None, stddev=False):
    n = len(algs)
    if labels is None:
        labels = alg_names
    table = ['-' for _ in range(n)]

    for i in range(n):
        if stddev:
            assert df[df.algo == algs[i]].rawloss.shape[0] == 10
            table[i] = '{:.3f} $\\pm$ {:.4f}'.format(
                    df[df.algo == algs[i]].rawloss.mean(),
                    df[df.algo == algs[i]].rawloss.std())
        else:
            assert df[df.algo == algs[i]].rawloss.shape[0] == 1
            table[i] = '{:.3f}'.format(df[df.algo == algs[i]].rawloss.mean())

    print(r'\begin{tabular}{ |', 'c | ' * n, '}')
    print(r'\hline')
    print(' & '.join(labels), r'\\ \hline')
    print(' & '.join(table), r'\\ \hline')
    print(r'\end{tabular}')


def print_loss_table_allnames(df, algs, labels=None, args=None):
    n = len(algs)
    if labels is None:
        labels = alg_names
    table = [['-' for _ in range(n)] for _ in range(4)]
    names = ['01', '01b', 'neg10', 'neg10b']
    col_labels = ['0/1', '0/1+b', '-1/0', '-1/0+b']

    for i in range(4):
        for j in range(n):
            assert df[df.algo == algs[j] + ':' + names[i]].rawloss.shape[0] == 1
            table[i][j] = '{:.3f}'.format(df[df.algo == algs[j] + ':' + names[i]].rawloss.mean())

    print(r'\begin{tabular}{ | c |', 'c | ' * n, '}')
    print(r'\hline')
    print(r' &', ' & '.join(labels), r'\\ \hline')
    for i in range(4):
        print(col_labels[i], '&', ' & '.join(table[i]), r'\\ \hline')
    print(r'\end{tabular}')


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
    parser.add_argument('--opt_name', action='store_true', default=False)
    parser.add_argument('--opt_algo', action='store_true', default=False)
    parser.add_argument('--comp_enc', action='store_true', default=False)
    parser.add_argument('--sep_cb_type', action='store_true', default=False)
    parser.add_argument('--sep_name', action='store_true', default=False)
    parser.add_argument('--sep_enc', action='store_true', default=False)
    parser.add_argument('--sep_b', action='store_true', default=False)
    parser.add_argument('--short', action='store_true', default=False,
                        help='only show main methods, for main paper')
    parser.add_argument('--skip_ips', action='store_true', default=False)
    parser.add_argument('--algo', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--enc', default=None)
    parser.add_argument('--b', default=None)
    parser.add_argument('--cb_type', default=None)
    parser.add_argument('--granular_ds', default=None)
    parser.add_argument('--granular_ds_name', default=None)
    parser.add_argument('--avg_std_name', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--use_cs', action='store_true', default=False)
    parser.add_argument('--use_hoeffding', action='store_true', default=False)
    parser.add_argument('--min_size', type=int, default=None)
    parser.add_argument('--max_size', type=int, default=None)
    parser.add_argument('--min_actions', type=int, default=None)
    parser.add_argument('--max_actions', type=int, default=None)
    parser.add_argument('--min_features', type=int, default=None)
    parser.add_argument('--max_features', type=int, default=None)
    parser.add_argument('--min_refloss', type=float, default=None)
    parser.add_argument('--max_refloss', type=float, default=None)
    parser.add_argument('--diff', action='store_true', default=True)
    parser.add_argument('--nodiff', dest='diff', action='store_false')
    parser.add_argument('--noval', action='store_true', default=False)
    parser.add_argument('--uci', action='store_true', default=False)
    parser.add_argument('--base_name', default='allrandfix')
    args = parser.parse_args()

    set_base_name(args.base_name)
    print((base_name()))

    if args.avg_std_name and args.base_name.startswith('rcv1'):
        names = ['{}01'.format(base_name())]
    else:
        names = ['{}{}'.format(base_name(), name) for name in ['01', '01b', 'neg10', 'neg10b']]
    df = load_names(names, use_cs=args.use_cs)

    # filters
    if args.min_actions is not None:
        df = df[df.na >= args.min_actions]
    if args.max_actions is not None:
        df = df[df.na <= args.max_actions]
    if args.min_features is not None:
        df = df[df.nf >= args.min_features]
    if args.max_features is not None:
        df = df[df.nf <= args.max_features]
    if args.min_size is not None:
        df = df[df.sz >= args.min_size]
    if args.max_size is not None:
        df = df[df.sz <= args.max_size]
    if args.min_refloss is not None:
        df = df[df.refloss >= args.min_refloss]
    if args.max_refloss is not None:
        df = df[df.refloss <= args.max_refloss]

    if args.noval:
        val_dss = np.load('ds_val_list.npy')
        df = df.loc[df.ds.map(lambda s: s not in val_dss)]
    if args.uci:
        uci_dss = ['6', '28', '30', '32', '54', '181', '182', '1590']
        df = df.loc[df.ds.map(lambda s: s in uci_dss)]
    print('num datasets:', len(df.ds.unique()))

    if (args.granular_ds_name or args.granular_name or args.avg_std_name) and args.name == 'neg10':
        # best fixed algos, selected on 200 datasets, -1/0 with no baseline
        g_best = 'epsilon:0:mtr'
        r_best = 'regcb:c0:0.001:mtr'
        ro_best = 'regcbopt:c0:0.001:mtr'
        cnu_best = 'cover:4:psi:0.1:nounif:dr'
        cu_best = 'cover:4:psi:0.1:ips'
        bg_best = 'bag:4:greedify:mtr'
        b_best = 'bag:4:mtr'
        eg_best = 'epsilon:0.02:mtr'
        a_best = 'epsilon:0.02:nounifa:c0:1e-06:mtr'
    elif (args.granular_ds_name or args.granular_name or args.avg_std_name) and args.name == '01':
        # best fixed algos, selected on 200 datasets, 0/1 with no baseline
        g_best = 'epsilon:0:mtr'
        r_best = 'regcb:c0:0.001:mtr'
        ro_best = 'regcbopt:c0:0.001:mtr'
        cnu_best = 'cover:4:psi:0.01:nounif:dr'
        cu_best = 'cover:4:psi:0.1:dr'
        bg_best = 'bag:8:greedify:mtr'
        b_best = 'bag:16:mtr'
        eg_best = 'epsilon:0.02:mtr'
        a_best = 'epsilon:0.02:nounifa:c0:1e-06:mtr'
    elif args.granular_neg10_bopt:
        # best fixed algos, selected on 200 datasets, -1/0 with optimized baseline
        g_best = 'epsilon:0:mtr'
        r_best = 'regcb:c0:0.001:mtr'
        ro_best = 'regcbopt:c0:0.001:mtr'
        cnu_best = 'cover:16:psi:0.1:nounif:dr'
        cu_best = 'cover:4:psi:0.1:ips'
        bg_best = 'bag:4:greedify:mtr'
        b_best = 'bag:4:mtr'
        eg_best = 'epsilon:0.02:mtr'
        a_best = 'epsilon:0.02:nounifa:c0:1e-06:mtr'

    psi = '0.1'
    if args.granular_opt or args.all:
        df_all = preprocess_df_granular(df, all_algos=True)

        # optimized name
        print('optimized over encoding/baseline')
        algs = ['epsilon:0:mtr', 'epsilon:0:dr', 'cover:16:psi:{}:nounif:dr'.format(psi),
                'bag:16:mtr', 'bag:16:greedify:mtr', 'epsilon:0.02:mtr', 'cover:16:psi:{}:dr'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular or args.all:
        print()
        print('best fixed encoding/baseline')
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        if args.short:
            algs = ['epsilon:0:mtr:neg10', 'regcbopt:c0:0.001:mtr:neg10',
                    'cover:4:psi:0.1:nounif:dr:neg10',
                    'bag:4:greedify:mtr:neg10b', 'epsilon:0.02:mtr:neg10']
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['epsilon:0:mtr:neg10',
                    'regcb:c0:0.001:mtr:neg10', 'regcbopt:c0:0.001:mtr:neg10b',
                    'cover:16:psi:0.1:nounif:dr:neg10',
                    'bag:4:mtr:neg10', 'bag:4:greedify:mtr:neg10b', 'epsilon:0.02:mtr:neg10',
                    'cover:8:psi:0.1:ips:neg10', 'epsilon:0.02:nounifa:c0:1e-06:mtr:neg10']
            labels = ['G', 'R', 'RO',
                       'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']  # 'e-d'
        print_table(df_all, algs, labels, args=args)

    if args.granular_neg10:
        print()
        print('fixed -1/0, fixed baseline choice (01 for active)')
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs = ['epsilon:0:mtr:neg10b', 'epsilon:0:dr:neg10b', 'cover:16:psi:{}:nounif:dr:neg10'.format(psi),
                'bag:16:mtr:neg10b', 'bag:16:greedify:mtr:neg10b', 'epsilon:0.02:mtr:neg10',
                'cover:16:psi:{}:dr:neg10'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr:01']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular_01:
        print()
        print('fixed 0/1, fixed baseline choice')
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs = ['epsilon:0:mtr:01b', 'epsilon:0:dr:01b', 'cover:16:psi:{}:nounif:dr:01'.format(psi),
                'bag:16:mtr:01b', 'bag:16:greedify:mtr:01b', 'epsilon:0.02:mtr:01',
                'cover:16:psi:{}:dr:01'.format(psi), 'epsilon:1:nounifa:c0:1e-06:dr:01']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu',
                  'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels, args=args)

    if args.granular_neg10_bopt or args.all:
        print()
        print('fixed -1/0, baseline optimized')
        df_all = preprocess_df_granular(df, all_algos=True, sep_enc=True)

        if args.short:
            algs = [g_best, ro_best, cnu_best, bg_best, eg_best]
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = [g_best, r_best, ro_best, cnu_best,
                    b_best, bg_best, eg_best, cu_best, a_best]
            labels = ['G', 'R', 'RO',
                       'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']  # 'e-d'
        for i in range(len(algs)):
            algs[i] += ':neg10'
        print_table(df_all, algs, labels, args=args)

    if args.granular_name or args.all:
        print()
        print('fixed name', args.name)
        assert args.name is not None, 'must specify --name'
        name = base_name() + args.name
        df_all = df.loc[df.name == name]
        df_all = preprocess_df_granular(df_all, all_algos=True)

        if args.short:
            algs = [g_best, ro_best, cnu_best, bg_best, eg_best]
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = [g_best, r_best, ro_best, cnu_best,
                    b_best, bg_best, eg_best, cu_best, a_best]
            labels = ['G', 'R', 'RO',
                       'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']  # 'e-d'

        print_table(df_all, algs, labels, args=args)

    if args.granular_ds:
        print()
        print('fixed ds', args.granular_ds)
        df_all = df.loc[df.ds == args.granular_ds]
        df_all = preprocess_df_granular(df_all, all_algos=True, sep_name=True)

        if args.short:
            algs = ['epsilon:0:mtr', 'regcbopt:c0:0.001:mtr',
                    'cover:16:psi:{}:nounif:dr'.format(psi),
                    'bag:16:greedify:mtr', 'epsilon:0.02:mtr']
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['epsilon:0:mtr', 'epsilon:0:dr',
                    'regcb:c0:0.001:mtr', 'regcbopt:c0:0.001:mtr',
                    'cover:16:psi:{}:nounif:dr'.format(psi),
                    'bag:16:mtr', 'bag:16:greedify:mtr', 'epsilon:0.02:mtr',
                    # 'cover:1:psi:{}:mtr'.format(psi),
                    'cover:16:psi:{}:dr'.format(psi), 'epsilon:0.02:nounifa:c0:1e-06:mtr']
            labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'R', 'RO',
                       'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']  # 'e-d'

        print_loss_table_allnames(df_all, algs, labels, args=args)

    if args.granular_ds_name:
        print()
        print('fixed ds', args.granular_ds, 'name', args.name)
        assert args.name is not None, 'must specify --name'
        name = base_name() + args.name
        df_all = df.loc[df.name == name]
        df_all = df_all.loc[df_all.ds == args.granular_ds_name]
        df_all = preprocess_df_granular(df_all, all_algos=True)

        if args.short:
            algs = [g_best, ro_best, cnu_best, bg_best, eg_best]
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = [g_best, r_best, ro_best, cnu_best,
                    b_best, bg_best, eg_best, cu_best, a_best]
            labels = ['G', 'R', 'RO',
                       'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']  # 'e-d'

        print_loss_table(df_all, algs, labels, args=args)

    if args.avg_std_name:
        print()
        print('mean +- std, fixed name', args.name)
        assert args.name is not None, 'must specify --name'
        name = base_name() + args.name
        df_all = df.loc[df.name == name]
        df_all = preprocess_df_granular(df_all, all_algos=True)

        if args.short:
            algs = [g_best, ro_best, cnu_best, bg_best, eg_best]
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = [g_best, r_best, ro_best, cnu_best,
                    b_best, bg_best, eg_best, cu_best, a_best]
            labels = ['G', 'R', 'RO',
                       'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']  # 'e-d'

        print_loss_table(df_all, algs, labels, args=args, stddev=True)

    if args.bag_vs_greedy or args.all:
        print()
        print('bag/bag-g vs greedy')
        df_all = preprocess_df_granular(df, all_algos=True, sep_name=True)

        algs_row = ['epsilon:0:mtr:neg10b', 'epsilon:0:dr:neg10b']
        labels_row = ['G-{}'.format(MTR_LABEL), 'G-dr']
        bag_algs = ['bag:{}:mtr:01b', 'bag:{}:greedify:mtr:01b',
                    'bag:{}:mtr:neg10b', 'bag:{}:greedify:mtr:neg10b']
        bag_labels = ['{}', '{}-g']
        print('0/1 + b')
        algs_col = [x.format(s) for s in ['4', '8', '16'] for x in bag_algs[:2]]
        labels_col = [x.format(s) for s in ['4', '8', '16'] for x in bag_labels]
        print_table_rect(df_all, algs_row, algs_col, labels_row, labels_col, args=args)
        print_table(df_all, algs_col[:2], labels_col[:2], args=args)
        print_table(df_all, algs_col[2:4], labels_col[2:4], args=args)
        print_table(df_all, algs_col[4:], labels_col[4:], args=args)

        print('-1/0 + b')
        algs_col = [x.format(s) for s in ['4', '8', '16'] for x in bag_algs[2:]]
        print_table_rect(df_all, algs_row, algs_col, labels_row, labels_col, args=args)
        print_table(df_all, algs_col[:2], labels_col[:2], args=args)
        print_table(df_all, algs_col[2:4], labels_col[2:4], args=args)
        print_table(df_all, algs_col[4:], labels_col[4:], args=args)

    if args.opt or args.all:
        print()
        print('optimize hyperparams, encoding/baseline')
        df_all = preprocess_df(df)

        if args.short:
            algs = ['greedy', 'regcbopt', 'cover_nounif', 'bag_greedy', 'e_greedy']
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['greedy', 'regcb', 'regcbopt', 'cover_nounif',
                    'bag', 'bag_greedy', 'e_greedy',
                    'cover', 'e_greedy_active']
            labels = ['G', 'R', 'RO', 'C-nu',
                      'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels=labels, args=args)

    if args.opt_neg10 or args.all:
        print()
        print('optimize hyperparams, encoding/baseline')
        # df_all = df.loc[df.cb_type != 'ips']
        df_all = preprocess_df(df, sep_enc=True, sep_b=(args.b is not None))
        # df_all = df_all.loc[(df_all.algo != 'greedy:neg10') | (df_all.cb_type != 'ips')]

        if args.short:
            algs = ['greedy', 'regcbopt', 'cover_nounif', 'bag_greedy', 'e_greedy']
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['greedy', 'regcb', 'regcbopt', 'cover_nounif',
                    'bag', 'bag_greedy', 'e_greedy',
                    'cover', 'e_greedy_active:01']
            labels = ['G', 'R', 'RO', 'C-nu',
                      'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        for i in range(len(algs)):
            algs[i] += ':neg10'
            if args.b is not None:
                algs[i] += ':' + args.b
        print_table(df_all, algs, labels=labels, args=args)

    if args.opt_01 or args.all:
        print()
        print('optimize hyperparams, encoding/baseline')
        # df_all = df.loc[df.cb_type != 'ips']
        df_all = preprocess_df(df, sep_enc=True, sep_b=(args.b is not None))
        # df_all = df_all.loc[(df_all.algo != 'greedy:neg10') | (df_all.cb_type != 'ips')]

        if args.short:
            algs = ['greedy', 'regcbopt', 'cover_nounif', 'bag_greedy', 'e_greedy']
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['greedy', 'regcb', 'regcbopt', 'cover_nounif',
                    'bag', 'bag_greedy', 'e_greedy',
                    'cover', 'e_greedy_active:01']
            labels = ['G', 'R', 'RO', 'C-nu',
                      'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        for i in range(len(algs)):
            algs[i] += ':01'
            if args.b is not None:
                algs[i] += ':' + args.b
        print_table(df_all, algs, labels=labels, args=args)

    if args.opt_name or args.all:
        print()
        print('optimize hyperparams', 'fixed name', args.name)
        assert args.name is not None, 'must specify --name'
        name = base_name() + args.name
        df_all = df.loc[df.name == name]
        # df_all = df_all.loc[df_all.cb_type != 'ips']
        df_all = preprocess_df(df_all)

        # df_all = df_all.loc[(df.algo != 'greedy') | (df.cb_type != 'ips')]

        if args.short:
            algs = ['greedy', 'regcbopt', 'cover_nounif', 'bag_greedy', 'e_greedy']
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['greedy', 'regcb', 'regcbopt', 'cover_nounif',
                    'bag', 'bag_greedy', 'e_greedy',
                    'cover', 'e_greedy_active:01']
            labels = ['G', 'R', 'RO', 'C-nu',
                      'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']
        print_table(df_all, algs, labels=labels, args=args)

    if False: # args.opt_01 or args.all:
        print()
        print('optimize hyperparams, encoding/baseline')
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
        print()
        print('optimize hyperparams, fixed encoding/baseline')
        df_all = preprocess_df(df, sep_name=args.sep_name or args.name,
                               sep_b=args.sep_b or args.b,
                               sep_enc=args.sep_enc or args.enc,
                               sep_reduction=args.sep_cb_type or args.cb_type)

        algo = args.algo
        algs = [algo]
        labels = None
        if args.sep_cb_type or args.cb_type:
            cb_types = [args.cb_type] if args.cb_type else ['ips', 'dr', 'mtr']
            if args.skip_ips:
                cb_types.remove('ips')
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

    if args.comp_enc:
        print()
        print('compare encodings, no baseline')
        assert args.min_size is None
        df_big = load_names(names, min_actions=args.min_actions, min_size=10000, use_cs=args.use_cs)
        df_all = preprocess_df(df, sep_b=True, sep_enc=True, sep_reduction=True)
        df_all_big = preprocess_df(df_big, sep_b=True, sep_enc=True, sep_reduction=True)

        if args.short:
            algs = ['greedy:mtr:{enc}:nb', 'regcbopt:mtr:{enc}:nb',
                    'cover_nounif:dr:{enc}:nb', 'bag_greedy:mtr:{enc}:nb',
                    'e_greedy:mtr:{enc}:nb'] 
            labels = ['G', 'RO', 'C-nu', 'B-g', r'$\epsilon$G']
        else:
            algs = ['greedy:mtr:{enc}:nb', 'greedy:dr:{enc}:nb', 'regcb:mtr:{enc}:nb', 'regcbopt:mtr:{enc}:nb',
                    'cover_nounif:dr:{enc}:nb', 'bag:mtr:{enc}:nb', 'bag_greedy:mtr:{enc}:nb',
                    'e_greedy:mtr:{enc}:nb', 'cover:dr:{enc}:nb', 'e_greedy_active:mtr:{enc}:nb'] 
            labels = ['G-iwr', 'G-dr', 'R', 'RO', 'C-nu', 'B', 'B-g', r'$\epsilon$G', 'C-u', 'A']

        print_enc_table(df_all, df_all_big, algs, labels, args=args)
