import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import kruskal,chi2_contingency,fisher_exact,spearmanr
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';rng=np.random.default_rng(2026)
pred=pd.read_csv(R/'predictions.csv');fold=pd.read_csv(R/'site_scores.csv');c=pd.read_csv(R/'analysis_cohort.csv')
metrics=['roc_auc','balanced_accuracy','f1','sensitivity','specificity'];B=10000
rows=[]
for (variant,mode,model),g in fold.groupby(['variant','feature_set','model'],sort=True):
 row={'variant':variant,'feature_set':mode,'model':model,'centers':len(g),'n':int(g.n.sum())};draw=rng.integers(0,len(g),size=(B,len(g)))
 for m in metrics:
  vals=g[m].to_numpy();valid=vals[np.isfinite(vals)];mdraw=rng.integers(0,len(valid),size=(B,len(valid)));boot=np.mean(valid[mdraw],axis=1);row['macro_'+m]=float(np.mean(valid));row[m+'_ci_low'],row[m+'_ci_high']=np.quantile(boot,[.025,.975]);row['valid_centers_'+m]=int(np.isfinite(vals).sum())
 pp=pred[(pred.variant==variant)&(pred.feature_set==mode)&(pred.model==model)]
 tn,fp,fn,tp=confusion_matrix(pp.y_true,pp.y_pred,labels=[0,1]).ravel()
 row.update(pooled_roc_auc=roc_auc_score(pp.y_true,pp.score),pooled_f1=f1_score(pp.y_true,pp.y_pred,zero_division=0),pooled_sensitivity=tp/(tp+fn),pooled_specificity=tn/(tn+fp),pooled_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp));rows.append(row)
summary=pd.DataFrame(rows);summary.to_csv(R/'summary_with_CI.csv',index=False)
# Conditional per-center class-stratified bootstrap, keeping both diagnostic groups.
cirows=[]
for (mode,model,center),g in pred[pred.variant=='primary'].groupby(['feature_set','model','center']):
 pos=g[g.y_true==1];neg=g[g.y_true==0];n1=len(pos);n0=len(neg);row=dict(feature_set=mode,model=model,center=center)
 if min(n1,n0)==0:continue
 wp=rng.multinomial(n1,np.full(n1,1/n1),size=2000);wn=rng.multinomial(n0,np.full(n0,1/n0),size=2000)
 mat=(pos.score.to_numpy()[:,None]>neg.score.to_numpy()[None,:]).astype(float)+.5*(pos.score.to_numpy()[:,None]==neg.score.to_numpy()[None,:])
 auc=np.einsum('bi,ij,bj->b',wp,mat,wn,optimize=True)/(n1*n0)
 tp=wp@pos.y_pred.to_numpy();fp=wn@neg.y_pred.to_numpy();sens=tp/n1;spec=1-fp/n0;f1=2*tp/(n1+tp+fp)
 for m,v in [('roc_auc',auc),('sensitivity',sens),('specificity',spec),('balanced_accuracy',(sens+spec)/2),('f1',f1)]:row[m+'_ci_low'],row[m+'_ci_high']=np.quantile(v,[.025,.975])
 cirows.append(row)
pd.DataFrame(cirows).to_csv(R/'per_center_CI.csv',index=False)

def paired(a,b,model,label):
 data=fold[(fold.variant=='primary')&(fold.model==model)].pivot(index='heldout_center',columns='feature_set',values='roc_auc')
 d=(data[a]-data[b]).dropna().to_numpy();n=len(d);boot=d[rng.integers(0,n,size=(B,n))].mean(axis=1)
 signs=np.array(list(itertools.product([-1,1],repeat=n)),dtype=np.int8)
 null=signs@d/n;p=float(np.mean(np.abs(null)>=abs(d.mean())-1e-12))
 return {'model':model,'comparison':label,'delta_macro_auc':float(d.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'p_signflip':p,'centers':n}

def holm(df,pcol):
 p=df[pcol].to_numpy();order=np.argsort(p);adj=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out=np.empty(len(p));out[order]=np.minimum(adj,1);return out
comparisons=pd.DataFrame([paired('combined','demographic',m,'combined_minus_demographic') for m in ['LR','SVM','RF']]);comparisons['p_holm']=holm(comparisons,'p_signflip');comparisons.to_csv(R/'primary_added_value.csv',index=False)
secondary=pd.DataFrame([paired('connectome','demographic',m,'connectome_minus_demographic') for m in ['LR','SVM','RF']]+[paired('combined','connectome',m,'combined_minus_connectome') for m in ['LR','SVM','RF']]);secondary['p_holm']=holm(secondary,'p_signflip');secondary.to_csv(R/'secondary_feature_comparisons.csv',index=False)
# Paired algorithm contrasts for each feature family; exploratory.
alg=[]
for mode in ['demographic','connectome','combined']:
 tab=fold[(fold.variant=='primary')&(fold.feature_set==mode)].pivot(index='heldout_center',columns='model',values='roc_auc')
 for a,b in [('SVM','LR'),('RF','LR'),('SVM','RF')]:
  d=(tab[a]-tab[b]).dropna().to_numpy();draw=d[rng.integers(0,len(d),size=(B,len(d)))].mean(axis=1);signs=np.array(list(itertools.product([-1,1],repeat=len(d))),dtype=np.int8)
  alg.append({'feature_set':mode,'comparison':a+'_minus_'+b,'delta_macro_auc':d.mean(),'ci_low':np.quantile(draw,.025),'ci_high':np.quantile(draw,.975),'p_signflip':np.mean(np.abs(signs@d/len(d))>=abs(d.mean())-1e-12)})
alg=pd.DataFrame(alg);alg['p_holm']=holm(alg,'p_signflip');alg.to_csv(R/'algorithm_comparisons.csv',index=False)
# Descriptive cohort audit, not outcome-guided filtering.
stats=[]
for (site,dx),g in c.groupby(['center','DX_GROUP']):
 stats.append({'center':site,'diagnosis':'ASD' if dx==1 else 'Control','n':len(g),'age_mean':g.AGE_AT_SCAN.mean(),'age_sd':g.AGE_AT_SCAN.std(),'age_min':g.AGE_AT_SCAN.min(),'age_max':g.AGE_AT_SCAN.max(),'female_n':int((g.SEX==2).sum()),'mean_FD':g.func_mean_fd.mean(),'FD_sd':g.func_mean_fd.std(),'frames_mean':g.n_frames.mean(),'missing_edge_n':int(g.has_missing_edges.sum()),'strict_manual_OK_n':int(g.strict_manual_ok.sum())})
pd.DataFrame(stats).to_csv(R/'cohort_by_center_and_diagnosis.csv',index=False)
qc=[]
for variable in ['AGE_AT_SCAN','func_mean_fd','n_frames','func_dvars','func_outlier']:
 values=[g[variable].dropna().to_numpy() for _,g in c.groupby('center')];stat,p=kruskal(*values);qc.append({'comparison':'between_centers','variable':variable,'test':'Kruskal-Wallis','statistic':stat,'p':p})
for variable in ['SEX','DX_GROUP','has_missing_edges']:
 tab=pd.crosstab(c.center,c[variable]);stat,p,_,_=chi2_contingency(tab);qc.append({'comparison':'between_centers','variable':variable,'test':'chi-square','statistic':stat,'p':p})
pd.DataFrame(qc).to_csv(R/'site_heterogeneity.csv',index=False)
# Investigate missingness and its observed mechanism, without inferring anatomical cause.
qcsource=pd.read_csv(R/'download_qc.csv');qcsource[qcsource.FILE_ID.isin(c.FILE_ID)].groupby('constant_roi_labels',dropna=False).agg(n=('FILE_ID','size'),missing_edges_min=('missing_edges','min'),missing_edges_max=('missing_edges','max')).to_csv(R/'constant_ROI_patterns.csv')
try:
 import statsmodels.api as sm
 import statsmodels.formula.api as smf
 cc=c.copy();cc['missing']=cc.has_missing_edges.astype(int);cc['ASD']=(cc.DX_GROUP==1).astype(int);cc['female']=(cc.SEX==2).astype(int);cc['FD_per_0_1mm']=cc.func_mean_fd/.1
 fit=smf.glm('missing ~ ASD + AGE_AT_SCAN + female + FD_per_0_1mm + C(center)',data=cc,family=sm.families.Binomial()).fit()
 tab=fit.summary2().tables[1];tab.to_csv(R/'missingness_GLM.csv');(R/'missingness_GLM.txt').write_text(fit.summary().as_text())
except Exception as e:(R/'missingness_GLM_error.txt').write_text(str(e))
# Verify all metrics and coverage independently from the saved participant predictions.
for _,row in fold.iterrows():
 pp=pred[(pred.variant==row.variant)&(pred.feature_set==row.feature_set)&(pred.model==row.model)&(pred.center==row.heldout_center)]
 assert pp.file_id.nunique()==len(pp)==row.n
 if len(np.unique(pp.y_true))==2:assert abs(roc_auc_score(pp.y_true,pp.score)-row.roc_auc)<1e-10
 assert confusion_matrix(pp.y_true,pp.y_pred,labels=[0,1]).ravel().tolist()==[row.tn,row.fp,row.fn,row.tp]
 assert abs(f1_score(pp.y_true,pp.y_pred,zero_division=0)-row.f1)<1e-10
 if pp.y_true.nunique()==2:
  sens=row.tp/(row.tp+row.fn);spec=row.tn/(row.tn+row.fp)
  assert abs(sens-row.sensitivity)<1e-10 and abs(spec-row.specificity)<1e-10
  assert abs((sens+spec)/2-row.balanced_accuracy)<1e-10
for key,g in pred.groupby(['variant','feature_set','model']):assert g.file_id.nunique()==len(g)
for a in json.load(open(R/'split_audit.json')):
 assert not set(a['train_ids'])&set(a['test_ids'])
 for inner in a['inner']:
  assert not set(inner['train_centers'])&set(inner['validation_centers']);assert a['heldout_center'] not in inner['train_centers']+inner['validation_centers'];assert not set(inner['train_ids']+inner['validation_ids'])&set(a['test_ids'])
(R/'verification.json').write_text(json.dumps({'status':'passed','conditions':len(summary),'predictions':len(pred),'outer_evaluations':len(fold),'bootstrap_site_resamples':10000,'per_center_resamples':2000,'uncertainty':'conditional on fitted CV predictions; overlapping training fits not accounted for; sign-flip tests exploratory'},indent=2))
print(summary[summary.variant=='primary'].to_string(index=False));print(comparisons.to_string(index=False));print('Statistical verification passed')
