# Contextual Bandit Bake-Off
Scripts for evaluation of contextual bandit algorithms in [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit).
The precise branch of VW used in the experiments is available [here](https://github.com/albietz/vowpal_wabbit/tree/bakeoff).
See the following paper for details:

A. Bietti, A. Agarwal, and J. Lanford. [A Contextual Bandit Bake-Off](https://arxiv.org/abs/1802.04064). arXiv preprint, 2018.

## Example usage

For making VW datasets, see `oml_to_vw.py` (for multiclass), `multilabel_to_vw.py` (for multilabel), `make_full.py`/`full_to_ldf.py` (for learning-to-rank).
Note that these require different VW options for simulating bandit feedback (`--cbify <num_actions>` for multiclass, `--cbify <num_actions> --cbify_cs` for multilabel/cost-sensitive, `--cbify_ldf` for datasets with label-dependent-features)

Here is an example bash script for running 100 jobs on multiclass datasets with -1/0 encoding:
```
name='resultsneg10'
for i in `seq 0 99`; do  # these should be run on different cores/machines with your own parallelization mechanism
  python2 run_vw_job.py $i 100 --name ${name} --flags '--loss0 -1 --loss1 0';
done;
```
