"""
Helper script for mslr/yahoo learning-to-rank datasets. To be used as follows (for 10 different shuffles):

    ### MSLR
    cat train.txt vali.txt test.txt | python make_full.py > train_full.txt
    for i in {1..10}; do shuf vw_full.txt > vw_full$i.txt; done
    for i in {1..10}; do cat vw_full$i.txt | python full_to_ldf.py | gzip > vw_full$i.vw.gz; done
    for i in {1..10}; do cp vw_full$i.vw.gz mslr_shufs/ds_mslr${i}_10.vw.gz; done

    ### Yahoo
    cat set*.txt | python make_full.py --max_docs 6 > vw_full.txt
    for i in {1..10}; do shuf vw_full.txt > vw_full$i.txt; done
    for i in {1..10}; do cat vw_full$i.txt | python full_to_ldf.py | gzip > vw_full$i.vw.gz; done
    for i in {1..10}; do cp vw_full$i.vw.gz yahoo_shufs/ds_yahoo${i}_6.vw.gz; done

note: for simulating bandit feedback, use the --cbify_ldf option in VW
"""

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_docs', type=int, default=10, help='max number of documents per query')
    args = parser.parse_args()

    curr_qid = None
    curr_num_docs = 0
    curr_line = ''

    for line in sys.stdin:
        rel, qid, features = line.strip().split(maxsplit=2)
        if curr_qid is None:
            curr_qid = qid
        if curr_qid != qid:
            print(curr_line)
            curr_line = ''
            curr_num_docs = 0
            curr_qid = qid

        if curr_num_docs >= args.max_docs:
            continue
        elif curr_num_docs > 0:
            curr_line += ','

        curr_num_docs += 1
        curr_line += '{}:{} | '.format(curr_num_docs, 1. - 0.25 * float(rel)) + features

    if curr_num_docs > 0:
        print(curr_line)
