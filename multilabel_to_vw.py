"""
Script for converting multi-label datasets to VW format.
The multi-label datasets in the original libsvm format can be found here:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html

note: for simulating bandit feedback, use the --cbify_cs <num_actions> option in VW
"""

import argparse
import gzip
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multilabel to vw')
    parser.add_argument('fname', help='input dataset file in libsvm format')
    parser.add_argument('dsname', help='name of the output VW dataset')
    parser.add_argument('na', type=int, help='number of actions/labels')
    args = parser.parse_args()

    outfile = 'ds_{}_{}.vw.gz'.format(args.dsname, args.na)
    with gzip.open(outfile, 'w') as f:
        for line in open(args.fname):
            label, rest = line.split(' ', 1)
            if label:
                labels = set(map(int, label.split(',')))
            else:
                labels = set()
            cs_label = ' '.join(
                    '{}:{}'.format(i + 1, '0.0' if i in labels else '1.0')
                    for i in range(args.na))
            f.write(cs_label + ' | ' + rest)
