import gzip
import pickle
import re
import sys

import numpy as np
import pandas as pd

def load_raw(loss_file, adf=True, cb_type=None, min_actions=None, min_size=None, shuffle=False):
    if adf:
        rgx = re.compile(r'^ds:(.+)\|na:(\d+)\|cb_type:(.*)\|(.*)\|(.*) (.*)$', flags=re.M)
        if loss_file.endswith('.gz'):
            lines = rgx.findall(gzip.open(loss_file).read().decode('utf-8'))
        else:
            lines = rgx.findall(open(loss_file).read())
        if cb_type == 'mtr':
            # filter cover for mtr since it's actually ips
            lines = [line for line in lines if 'cover' not in line[4]]
        if cb_type is None:  # append cb_type to algo name
            lines = [list(line) for line in lines]
            for line in lines:
                if line[4] != 'supervised':
                    line[4] += ':' + line[2]
        df_raw = pd.DataFrame(lines, columns=['ds', 'na', 'cb_type', 'lr', 'algo', 'loss'])
        if cb_type is not None:
            df_raw = df_raw[df_raw.cb_type == cb_type]
    else:
        # rgx = re.compile('^ds_(\d+)_(\d+)_(.*) (.*)$', flags=re.M)
        rgx = re.compile(r'^ds:(.+)\|na:(\d+)\|(.*)\|(.*) (.*)$', flags=re.M)
        lines = rgx.findall(open(loss_file).read())

        df_raw = pd.DataFrame(lines, columns=['ds', 'na', 'lr', 'algo', 'loss'])
    # df_raw.ds = df_raw.ds.astype(int)
    df_raw.na = df_raw.na.astype(int)
    df_raw.loss = df_raw.loss.astype(float)
    df_raw.algo = df_raw.algo.map(lambda x:
            x.replace('nounifagree', 'nounifa')
            .replace('agree_mellowness', 'c0')
            .replace('mellowness', 'c0')
            )

    ds_to_sz = pickle.load(open('ds_sz.pkl', 'rb'), encoding='latin1')
    ds_to_nf = pickle.load(open('ds_nf.pkl', 'rb'), encoding='latin1')
    df_raw['sz'] = df_raw.ds.map(ds_to_sz)
    df_raw['nf'] = df_raw.ds.map(ds_to_nf)

    if min_actions is not None:
        df_raw = df_raw[df_raw.na >= min_actions]
    if min_size is not None:
        df_raw = df_raw[df_raw.sz >= min_size]

    if shuffle:
        df_raw.sample(frac=1)

    return df_raw
