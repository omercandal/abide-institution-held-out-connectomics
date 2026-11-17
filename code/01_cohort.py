import json,io,time,hashlib,urllib.request,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';R=ROOT/'results';CACHE=D/'roi_cache';CACHE.mkdir(exist_ok=True,parents=True)
URL='https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/'
with urllib.request.urlopen(URL+'Phenotypic_V1_0b_preprocessed1.csv',timeout=60) as r:raw=r.read()
(D/'phenotypic_source.csv').write_bytes(raw)
p=pd.read_csv(io.BytesIO(raw));p.columns=p.columns.str.strip()
for c in ['DX_GROUP','AGE_AT_SCAN','SEX','func_mean_fd']:p[c]=pd.to_numeric(p[c],errors='coerce')
p.loc[p.AGE_AT_SCAN<=0,'AGE_AT_SCAN']=np.nan
p.loc[~p.SEX.isin([1,2]),'SEX']=np.nan
assert p.SUB_ID.is_unique
p['reason']='eligible_metadata'
for mask,reason in [(~p.DX_GROUP.isin([1,2]),'invalid_diagnosis'),(p.FILE_ID.isna()|p.FILE_ID.eq('no_filename'),'no_preprocessed_filename'),(~p.func_mean_fd.between(0,.3),'missing_invalid_or_FD_over_0.30')]:p.loc[(p.reason=='eligible_metadata')&mask,'reason']=reason
qcols=['qc_rater_1','qc_anat_rater_2','qc_func_rater_2','qc_anat_rater_3','qc_func_rater_3']
p['manual_fail']=p[qcols].apply(lambda s:s.str.lower().eq('fail')).any(axis=1)
p['strict_manual_ok']=p[qcols].apply(lambda s:s.str.lower().eq('ok')).all(axis=1)
elig=p[p.reason=='eligible_metadata'].copy();print('Metadata eligible for download',len(elig),flush=True)
# Reuse only publicly retrieved pilot scans, verifying every scan again.
pilot=ROOT.parent/'abide_analysis_outputs'/'roi_cache'
for fid in elig.FILE_ID:
 src=pilot/f'{fid}_rois_ho.1D'
 if src.exists():shutil.copy(src,CACHE/src.name)
def get(row):
 fid=row.FILE_ID;path=CACHE/f'{fid}_rois_ho.1D';url=URL+f'Outputs/cpac/filt_noglobal/rois_ho/{fid}_rois_ho.1D'
 result={'FILE_ID':fid,'url':url,'error':'','n_frames':0,'n_roi':0,'missing_edges':0,'constant_roi_labels':'','sha256':''}
 for attempt in range(3):
  try:
   if not path.exists():
    with urllib.request.urlopen(url,timeout=45) as response:path.write_bytes(response.read())
   b=path.read_bytes();a=np.loadtxt(io.BytesIO(b),comments='#',ndmin=2);labels=b.decode().splitlines()[0].split()
   result.update(n_frames=a.shape[0],n_roi=a.shape[1],sha256=hashlib.sha256(b).hexdigest())
   if a.shape[1]!=111:raise ValueError('unexpected_ROI_count')
   if a.shape[0]<70:raise ValueError('fewer_than_70_frames')
   if not np.isfinite(a).all():raise ValueError('nonfinite_time_series')
   sd=a.std(axis=0);bad=sd<=1e-12
   result['constant_roi_labels']=';'.join(np.asarray(labels)[bad])
   with np.errstate(divide='ignore',invalid='ignore'):corr=np.corrcoef(a,rowvar=False)
   corr[bad,:]=np.nan;corr[:,bad]=np.nan
   edges=np.arctanh(np.clip(corr[np.triu_indices(111,1)],-.999999,.999999)).astype('float32')
   if not np.isfinite(edges).any():raise ValueError('no_finite_edges')
   result['missing_edges']=int(np.isnan(edges).sum());result['edges']=edges;result['labels']=labels
   return result
  except Exception as e:
   result['error']=str(e)
   if isinstance(e,ValueError):break
   time.sleep(.4*(attempt+1))
 return result
res=[]
with ThreadPoolExecutor(max_workers=12) as pool:
 jobs=[pool.submit(get,r) for r in elig.itertuples(index=False)]
 for fut in as_completed(jobs):
  x=fut.result()
  if 'edges' in x:x['error']=''
  res.append(x)
  if len(res)%50==0 or len(res)==len(jobs):print('Scans processed',len(res),'/',len(jobs),flush=True)
qc=pd.DataFrame([{k:v for k,v in x.items() if k not in ['edges','labels']} for x in res]).sort_values('FILE_ID');qc.to_csv(R/'download_qc.csv',index=False)
valid={x['FILE_ID']:x for x in res if not x['error']}
assert len({tuple(x['labels']) for x in valid.values()})==1
labels=next(iter(valid.values()))['labels'];(D/'roi_labels.json').write_text(json.dumps(labels))
p=p.merge(qc.drop(columns=['url']),on='FILE_ID',how='left',validate='many_to_one')
p.loc[(p.reason=='eligible_metadata')&p.manual_fail,'reason']='manual_quality_fail'
p.loc[(p.reason=='eligible_metadata')&~p.FILE_ID.isin(valid),'reason']='download_or_timeseries_QC_failure'
p.to_csv(R/'all_participants_selection.csv',index=False)
c=p[p.reason=='eligible_metadata'].sort_values(['SITE_ID','SUB_ID']).copy()
# Do not exclude small sites on the basis of their observed model performance.
assert c.SUB_ID.is_unique and c.FILE_ID.is_unique
Xc=np.vstack([valid[fid]['edges'] for fid in c.FILE_ID]);Xd=c[['AGE_AT_SCAN','SEX','func_mean_fd']].to_numpy(float)
c['has_missing_edges']=np.isnan(Xc).any(axis=1);c.to_csv(R/'analysis_cohort.csv',index=False)
np.savez_compressed(D/'features.npz',Xc=Xc,Xd=Xd,y=(c.DX_GROUP.to_numpy()==1).astype(int),sites=c.SITE_ID.to_numpy(dtype=str),ids=c.FILE_ID.to_numpy(dtype=str),subject_ids=c.SUB_ID.to_numpy())
flow={'phenotypic_n':len(p),'metadata_download_n':len(elig),'primary_n':len(c),'sequential_reasons':p.reason.value_counts().to_dict(),'sites':int(c.SITE_ID.nunique()),'classes':c.DX_GROUP.value_counts().to_dict(),'missing_edge_subjects':int(c.has_missing_edges.sum()),'metadata_sha256':hashlib.sha256(raw).hexdigest()}
(R/'cohort_flow.json').write_text(json.dumps(flow,indent=2));print(json.dumps(flow,indent=2),flush=True)
