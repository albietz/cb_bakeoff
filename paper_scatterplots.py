import matplotlib
matplotlib.use('Agg')
import argparse
from eval_loss import load_names
from rank_algos import significance, significance_cs01, preprocess_df_granular, preprocess_df, base_name, set_base_name
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('ggplot')

FIGDIR = '/scratch/clear/abietti/cb_eval/plots/scatter/'
MTR_LABEL = 'iwr'

def scatterplot(df, alg_names, labels=None,
                lim_min=-0.25, lim_max=1., args=None, fname=None):
    assert len(alg_names) == 2
    if labels is None:
        labels = alg_names

    rawx = df.loc[df.algo == alg_names[0]].groupby('ds').rawloss.mean()
    rawy = df.loc[df.algo == alg_names[1]].groupby('ds').rawloss.mean()
    if args.rawloss:
        x, y = rawx, rawy
    else:
        x = df.loc[df.algo == alg_names[0]].groupby('ds').loss.mean()
        y = df.loc[df.algo == alg_names[1]].groupby('ds').loss.mean()
    sz = df.loc[df.algo == alg_names[0]].groupby('ds').sz.max()
    if args.use_cs:
        pvals = significance_cs01(rawx, rawy, sz)
    else:
        pvals = significance(rawx, rawy, sz)

    plt.figure(figsize=(2.5,2.5))
    # plt.scatter(x, y,
    #             s=plt.rcParams['lines.markersize']**2 * (pvals < args.alpha).map(lambda x: 0.7 if x else 0.2),
    #             c=(pvals < args.alpha).map(lambda x: 'r' if x else 'k'))
    sign_idxs = (pvals < args.alpha)
    nsign_idxs = np.logical_not(sign_idxs)
    plt.scatter(x[nsign_idxs], y[nsign_idxs], s=plt.rcParams['lines.markersize']**2 * 0.2, c='k')
    plt.scatter(x[sign_idxs], y[sign_idxs], s=plt.rcParams['lines.markersize']**2 * 0.7, c='r')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color='k')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if fname is not None:
        figname = fname + ('_cs' if args.use_cs else '')
    else:
        figname = '_vs_'.join(alg_names).replace(':', '_').replace('.', '')
    if args.min_actions is not None:
        figname += '_{}a'.format(args.min_actions)
    figname += '.pdf'
    plt.savefig(os.path.join(FIGDIR, figname), #'{}_{}'.format(base_name(), figname)),
                bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scatterplots')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--enc_910', action='store_true', default=False)
    parser.add_argument('--comp_granular', action='store_true', default=False)
    parser.add_argument('--comp_granular_01', action='store_true', default=False)
    parser.add_argument('--bag_greedy', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument('--cover_nu', action='store_true', default=False)
    parser.add_argument('--active', action='store_true', default=False)
    parser.add_argument('--active_mtr', action='store_true', default=False)
    parser.add_argument('--regcb', action='store_true', default=False)
    parser.add_argument('--regcb_cover', action='store_true', default=False)
    parser.add_argument('--greedy', action='store_true', default=False)
    parser.add_argument('--psi', default='0.1')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--use_cs', action='store_true', default=False)
    parser.add_argument('--base_name', default='allrandfix')
    parser.add_argument('--rawloss', action='store_true', default=False)
    parser.add_argument('--introlabels', action='store_true', default=False)
    parser.add_argument('--min_actions', type=int, default=None)
    parser.add_argument('--noval', action='store_true', default=False)
    args = parser.parse_args()

    set_base_name(args.base_name)

    if args.noval:
        val_dss = np.load('ds_val_list.npy')

    def filter_heldout(df):
        if args.noval:
            return df.loc[df.ds.map(lambda s: s not in val_dss)]
        else:
            return df

    if args.enc_910 or args.all:
        names = ['disagree910', 'disagree910b']
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        df = filter_heldout(df)
        algs = ['epsilon:0:mtr', 'cover:4:psi:0.1:nounif:dr', 'bag:4:greedify:mtr']
        labels = ['G', 'C-nu', 'B-g']
        df = preprocess_df_granular(df, algos=algs, sep_name=True)

        for i in range(3):
            scatterplot(df,
                [algs[i] + ':910', algs[i] + ':910b'],
                [labels[i] + ', 9/10', labels[i] + ', 9/10 + b'],
                args=args, fname='robust910_' + labels[i].lower())

    if args.comp_granular or args.all:
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=args.min_actions, use_cs=args.use_cs)
        df = filter_heldout(df)
        # df = df.loc[df.na >= 5]
        algs = ['epsilon:0:mtr:neg10',
                'cover:4:psi:0.1:nounif:dr:neg10'.format(args.psi), 'bag:4:greedify:mtr:neg10',
                'regcbopt:c0:0.001:mtr:neg10']
        # algs = ['epsilon:0:mtr:neg10', 'epsilon:0:dr:neg10',
        #         'cover:16:psi:{}:nounif:dr:neg10'.format(args.psi), 'bag:16:greedify:mtr:neg10']
        labels = ['G', 'C-nu', 'B-g', 'RO']
        if args.introlabels:
            labels = ['greedy loss', 'cover-nu loss', 'bag-g loss', 'regcb-opt loss']
        df = preprocess_df_granular(df, algos=[a[:a.rfind(':')] for a in algs], sep_name=True)

        # for i, j in [(0, 1), (1, 2), (2, 3), (1, 3)]:
        for i, j in [(0, 1), (1, 2), (0, 2), (0, 3), (3, 1), (3, 2)]:
            scatterplot(df,
                [algs[i], algs[j]],
                [labels[i], labels[j]],
                args=args,
                fname='comp_' + labels[i].lower().replace(' ', '_') + '_' + labels[j].lower().replace(' ', '_'))

    if args.comp_granular_01 or args.all:
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        # df = df.loc[df.na >= 5]
        algs = ['epsilon:0:mtr:01b', 'epsilon:0:dr:01b',
                'cover:16:psi:{}:nounif:dr:01'.format(args.psi), 'bag:16:greedify:mtr:01b']
        # algs = ['epsilon:0:mtr:neg10', 'epsilon:0:dr:neg10',
        #         'cover:16:psi:{}:nounif:dr:neg10'.format(args.psi), 'bag:16:greedify:mtr:neg10']
        labels = ['G-{}'.format(MTR_LABEL), 'G-dr', 'C-nu', 'B-g']
        # labels = ['G-{}'.format(MTR_LABEL), 'greedy loss', 'cover loss', 'bag']
        df = preprocess_df_granular(df, algos=[a[:a.rfind(':')] for a in algs], sep_name=True)

        for i, j in [(0, 1), (1, 2), (2, 3), (1, 3)]:
            scatterplot(df,
                [algs[i], algs[j]],
                [labels[i], labels[j]],
                args=args, fname='comp01_' + labels[i].lower() + '_' + labels[j].lower())

    if args.bag_greedy or args.all:
        names = [base_name() + 'neg10b']
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        algs = ['bag:{}:mtr', 'bag:{}:greedify:mtr']
        labels = ['bag {}', 'bag-g {}']

        for s in ['4', '8', '16']:
            scatterplot(df,
                [algs[0].format(s), algs[1].format(s)],
                [labels[0].format(s), labels[1].format(s)],
                args=args, fname='bag_greedy_neg10b_' + s)

    if args.cover_nu or args.all:
        names = [base_name() + 'neg10']
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        algs = ['cover:{}:psi:0.1:dr', 'cover:{}:psi:0.1:nounif:dr']
        labels = ['C-u', 'C-nu']

        for s in ['4', '8', '16']:
            scatterplot(df,
                [algs[1].format(s), algs[0].format(s)],
                [labels[1].format(), labels[0].format()],
                args=args, fname='cover_nu_neg10_' + s)

    if args.baseline or args.all:
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        df = filter_heldout(df)
        algs = ['epsilon:0:mtr', 'regcbopt:c0:0.001:mtr', 'cover:4:psi:0.1:nounif:dr', 'bag:4:greedify:mtr']
        labels = ['G', 'RO', 'C-nu', 'B-g']
        df = preprocess_df_granular(df, algos=algs, sep_name=True)

        for i in range(4):
            scatterplot(df,
                [algs[i] + ':01', algs[i] + ':01b'],
                [labels[i] + ' 0/1', labels[i] + ' 0/1 + b'],
                args=args, fname='baseline_' + labels[i].lower() + '_01')
            scatterplot(df,
                [algs[i] + ':neg10', algs[i] + ':neg10b'],
                [labels[i] + ' -1/0', labels[i] + ' -1/0 + b'],
                args=args, fname='baseline_' + labels[i].lower() + '_neg10')

    if args.active or args.all:
        # names = ['disagree01']
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        df = preprocess_df_granular(
                df,
                all_algos=True, sep_name=True)
        algs = ['epsilon:0.02:dr:01', 'epsilon:0.02:nounifa:c0:{}:dr:01']
        labels = ['$\epsilon$ = 0.02', '$\epsilon$ = 0.02, $C_0$ = {}']

        for c0, c0_lab in [('0.01', '1e-2'), ('0.0001', '1e-4'), ('1e-06', '1e-6')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='active_dr01_' + c0_lab)

        algs = ['epsilon:0.02:dr:neg10', 'epsilon:0.02:nounifa:c0:{}:dr:neg10']
        labels = ['$\epsilon$ = 0.02', '$\epsilon$ = 0.02, $C_0$ = {}']

        for c0, c0_lab in [('0.01', '1e-2'), ('0.0001', '1e-4'), ('1e-06', '1e-6')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='active_drn10_' + c0_lab)

        algs = ['epsilon:0.02:mtr:01', 'epsilon:0.02:nounifa:c0:{}:mtr:01']
        labels = ['$\epsilon$ = 0.02', '$\epsilon$ = 0.02, $C_0$ = {}']

        for c0, c0_lab in [('0.01', '1e-2'), ('0.0001', '1e-4'), ('1e-06', '1e-6')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='active_mtr01_' + c0_lab)

        algs = ['epsilon:0.02:mtr:neg10', 'epsilon:0.02:nounifa:c0:{}:mtr:neg10']
        labels = ['$\epsilon$ = 0.02', '$\epsilon$ = 0.02, $C_0$ = {}']

        for c0, c0_lab in [('0.01', '1e-2'), ('0.0001', '1e-4'), ('1e-06', '1e-6')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='active_mtrn10_' + c0_lab)

    if args.regcb:
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        df = filter_heldout(df)
        df = preprocess_df_granular(
                df,
                all_algos=True, sep_name=True)
        algs = ['epsilon:0:mtr:neg10', 'regcb:c0:{}:mtr:neg10']
        labels = ['G', 'R, $C_0$ = {}']

        for c0, c0_lab in [('0.1', '1e-1'), ('0.01', '1e-2'), ('0.001', '1e-3')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='regcb_' + c0_lab)

        algs = ['epsilon:0:mtr:neg10', 'regcbopt:c0:{}:mtr:neg10']
        labels = ['G', 'RO, $C_0$ = {}']

        for c0, c0_lab in [('0.1', '1e-1'), ('0.01', '1e-2'), ('0.001', '1e-3')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='regcbopt_' + c0_lab)

    if args.regcb_cover:
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        df = filter_heldout(df)
        df = preprocess_df_granular(
                df,
                all_algos=True, sep_name=True)
        algs = ['cover:4:psi:0.1:nounif:dr:neg10', 'regcb:c0:{}:mtr:neg10']
        labels = ['C-nu', 'R, $C_0$ = {}']

        for c0, c0_lab in [('0.1', '1e-1'), ('0.01', '1e-2'), ('0.001', '1e-3')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='regcb_cnu_' + c0_lab)

        algs = ['cover:16:psi:0.1:nounif:dr:neg10', 'regcbopt:c0:{}:mtr:neg10']
        labels = ['C-nu', 'RO, $C_0$ = {}']

        for c0, c0_lab in [('0.1', '1e-1'), ('0.01', '1e-2'), ('0.001', '1e-3')]:
            scatterplot(df,
                [algs[1].format(c0), algs[0]],
                [labels[1].format(c0_lab), labels[0]],
                args=args, fname='regcbopt_cnu_' + c0_lab)

    if args.greedy or args.all:
        names = [base_name() + name for name in ['01', '01b', 'neg10', 'neg10b']]
        df = load_names(names, min_actions=None, use_cs=args.use_cs)
        algs = ['epsilon:0:mtr:neg10', 'epsilon:{}:mtr:neg10']
        # algs = ['epsilon:0:mtr:neg10', 'epsilon:{}:mtr:neg10']
        labels = ['G', '$\epsilon$G']
        epsilons = ['0', '0.02', '0.05', '0.1']
        df = preprocess_df_granular(
                df,
                algos=['epsilon:{}:mtr'.format(eps) for eps in epsilons], sep_name=True)

        for eps in epsilons[1:]:
            scatterplot(df,
                [algs[0], algs[1].format(eps)],
                [labels[0], labels[1].format(eps)],
                args=args, fname='greedy_' + eps.replace('.', ''))
