import argparse
from config import OML_API_KEY
import gzip
import openml
import os
import scipy.sparse as sp

VW_DS_DIR = '/bscratch/b-albiet/vwdatasets/'

def save_vw_dataset(X, y, did, ds_dir):
    n_classes = y.max() + 1
    fname = 'ds_{}_{}.vw.gz'.format(did, n_classes)
    with gzip.open(os.path.join(ds_dir, fname), 'w') as f:
        if sp.isspmatrix_csr(X):
            for i in range(X.shape[0]):
                f.write('{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(X[i].indices, X[i].data))))
        else:
            for i in range(X.shape[0]):
                f.write('{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in enumerate(X[i]) if val != 0)))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openML to vw converter')
    parser.add_argument('min_did', type=int, default=0, help='minimum dataset id to process')
    parser.add_argument('max_did', type=int, default=None, help='maximum dataset id to process')
    args = parser.parse_args()
    print args.min_did, ' to ', args.max_did

    openml.config.apikey = OML_API_KEY
    openml.config.set_cache_directory('/scratch/clear/abietti/cb_eval/omlcache/')

    print 'loaded openML'

    if not os.path.exists(VW_DS_DIR):
        os.makedirs(VW_DS_DIR)

    dids = [int(x) for x in os.listdir('/bscratch/b-albiet/omlcache/datasets/')]

    # min_did = 1200
    # max_did = 1500
    for did in sorted(dids):
        if did < args.min_did:
            continue
        if args.max_did is not None and did >= args.max_did:
            break
        print 'processing did', did
        try:
            ds = openml.datasets.get_dataset(did)
            X, y = ds.get_data(target=ds.default_target_attribute)
        except Exception as e:
            print e
            continue
        save_vw_dataset(X, y, did, VW_DS_DIR)
