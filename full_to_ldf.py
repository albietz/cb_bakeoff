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

import sys

if __name__ == '__main__':
    for line in sys.stdin:
        for lab_line in line.strip().split(','):
            print(lab_line)
        print()
