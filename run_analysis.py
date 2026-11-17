"""Reconstruct analysis locally after explicit source-data retrieval authorization."""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VARIANTS=['primary','complete_case','drop_roi3455','low_motion','pca8','strict_manual','residualized']
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fetch-public-data',action='store_true',help='Retrieve public PCP source derivatives after reviewing source terms')
    ap.add_argument('--resume',action='store_true',help='Allow recovery of existing local fold checkpoints; this is not a fresh independent refit')
    args=ap.parse_args()
    for name in ['data','results','checkpoints','figures']:(ROOT/name).mkdir(exist_ok=True)
    if any((ROOT/'checkpoints').glob('*.json')) and not args.resume:
        ap.error('Existing checkpoints found. Use a new working copy for a fresh refit, or explicitly request --resume.')
    if not (ROOT/'data/features.npz').exists() and not args.fetch_public_data:
        ap.error('No participant features present. Read the data terms and use --fetch-public-data to reconstruct from the provider.')
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(script,extra=None):
        settings=env.copy();settings.update(extra or {})
        subprocess.run([sys.executable,'-u',str(ROOT/'code'/script)],cwd=ROOT,env=settings,check=True)
    if not (ROOT/'data/features.npz').exists():run('01_cohort.py')
    for variant in VARIANTS:run('02_models.py',{'ABIDE_VARIANT':variant})
    import pandas as pd
    for stem in ['predictions','site_scores','inner_search']:
        pd.concat([pd.read_csv(ROOT/'results'/f'{stem}_{v}.csv') for v in VARIANTS],ignore_index=True).to_csv(ROOT/'results'/f'{stem}.csv',index=False)
    audit=[]
    for variant in VARIANTS:audit.extend(json.loads((ROOT/'results'/f'split_audit_{variant}.json').read_text()))
    (ROOT/'results/split_audit.json').write_text(json.dumps(audit))
    run('03_statistics.py');run('04_figures.py')
    print('Reconstructed outputs saved in local, git-ignored results/ and figures/.')
if __name__=='__main__':main()
