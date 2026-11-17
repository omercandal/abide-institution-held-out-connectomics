"""Verify public aggregate outputs without loading participant-level data."""
from pathlib import Path
import csv,hashlib,json,math,statistics

ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/'aggregate_results'/name).open(newline='') as f:return list(csv.DictReader(f))
def number(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')
def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for item in manifest['files']:
        p=ROOT/item['path']
        assert p.is_file(),f'Missing {p}'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],f'Hash mismatch {p}'
    summary=read('summary_with_CI.csv');sites=read('site_scores.csv')
    checks=0
    for row in summary:
        selected=[x for x in sites if (x['variant'],x['feature_set'],x['model'])==(row['variant'],row['feature_set'],row['model'])]
        assert len(selected)==int(row['centers'])
        assert sum(int(x['n']) for x in selected)==int(row['n'])
        for metric in ['roc_auc','balanced_accuracy','f1','sensitivity','specificity']:
            values=[number(x[metric]) for x in selected];values=[v for v in values if math.isfinite(v)]
            assert len(values)==int(row['valid_centers_'+metric])
            assert math.isclose(statistics.mean(values),float(row['macro_'+metric]),abs_tol=1e-12)
            checks+=1
    for row in read('primary_added_value.csv'):
        sums={x['feature_set']:float(x['macro_roc_auc']) for x in summary if x['variant']=='primary' and x['model']==row['model']}
        assert math.isclose(sums['combined']-sums['demographic'],float(row['delta_macro_auc']),abs_tol=1e-12)
        checks+=1
    cohort=read('cohort_by_center_and_diagnosis.csv')
    assert sum(int(x['n']) for x in cohort)==830
    assert sum(int(x['n']) for x in cohort if x['diagnosis']=='ASD')==376
    assert len({x['center'] for x in cohort})==17
    assert len(summary)==37 and len(sites)==626
    print(json.dumps({'status':'passed','files_hashed':len(manifest['files']),'aggregate_metric_checks':checks,'conditions':37,'outer_evaluations':626,'primary_n':830,'primary_ASD':376,'institution_groups':17,'models_refitted':False},indent=2))
if __name__=='__main__':main()
