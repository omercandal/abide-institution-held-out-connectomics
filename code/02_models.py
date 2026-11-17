import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,itertools,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score,balanced_accuracy_score
from sklearn.base import clone
import sklearn
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';C=ROOT/'checkpoints';SEED=2026
z=np.load(ROOT/'data/features.npz');cohort=pd.read_csv(R/'analysis_cohort.csv');Xc=z['Xc'];Xd=z['Xd'];y=z['y'];ids=z['ids'];rawsites=z['sites']
sites=np.array([s.rsplit('_',1)[0] if s in ['UM_1','UM_2','UCLA_1','UCLA_2','LEUVEN_1','LEUVEN_2'] else s for s in rawsites]);cohort['center']=sites;cohort.to_csv(R/'analysis_cohort.csv',index=False)
labels=json.load(open(ROOT/'data/roi_labels.json'));ij=np.triu_indices(111,1);bad=labels.index('#3455');keep=(ij[0]!=bad)&(ij[1]!=bad)

def measure(yt,score,pred):
 tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
 return {'n':len(yt),'roc_auc':float(roc_auc_score(yt,score)) if len(np.unique(yt))==2 else None,'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2) if min(tp+fn,tn+fp)>0 else None,'f1':float(f1_score(yt,pred,zero_division=0)),'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def grids(name):
 if name=='LR':return [{'C':v} for v in [.1,1.,10.]]
 if name=='SVM':return [{'C':v,'kernel':k} for v in [.1,1.,10.] for k in ['linear','rbf']]
 return [{'max_depth':v} for v in [5,None]]
def estimator(name,params):
 if name=='LR':return LogisticRegression(**params,solver='liblinear',class_weight='balanced',random_state=SEED,max_iter=5000)
 if name=='SVM':return SVC(**params,class_weight='balanced',probability=False,random_state=SEED)
 return RandomForestClassifier(**params,n_estimators=200,min_samples_leaf=3,class_weight='balanced',random_state=SEED,n_jobs=1)

def prepare(train,test,fc,ncomp,residual=False):
 # This function is called separately for every inner and outer training split.
 di=SimpleImputer(strategy='median',keep_empty_features=True);ds=StandardScaler()
 dt=di.fit_transform(Xd[train]);dv=di.transform(Xd[test]);dtr=ds.fit_transform(dt);dte=ds.transform(dv)
 fi=SimpleImputer(strategy='median',keep_empty_features=True)
 ft=fi.fit_transform(fc[train]);fv=fi.transform(fc[test])
 if residual:
  reg=LinearRegression();reg.fit(dtr,ft);ft=ft-reg.predict(dtr);fv=fv-reg.predict(dte)
 fs=StandardScaler();ft=fs.fit_transform(ft);fv=fs.transform(fv)
 pc=PCA(n_components=ncomp,svd_solver='randomized',random_state=SEED);ft=pc.fit_transform(ft);fv=pc.transform(fv)
 cs=StandardScaler();ft=cs.fit_transform(ft);fv=cs.transform(fv)
 return {'demographic':(dtr,dte),'connectome':(ft,fv),'combined':(np.column_stack([dtr,ft]),np.column_stack([dte,fv]))},float(pc.explained_variance_ratio_.sum())
variants=[('primary',np.ones(len(y),bool),Xc,32,['LR','SVM','RF'],False),('complete_case',~np.isnan(Xc).any(axis=1),Xc,32,['LR','SVM','RF'],False),('drop_roi3455',np.ones(len(y),bool),Xc[:,keep],32,['LR','SVM','RF'],False),('low_motion',Xd[:,2]<=.2,Xc,32,['LR'],False),('pca8',np.ones(len(y),bool),Xc,8,['LR'],False),('strict_manual',cohort.strict_manual_ok.to_numpy(bool),Xc,32,['LR'],False),('residualized',np.ones(len(y),bool),Xc,32,['LR'],True)]
selected_variant=os.environ.get('ABIDE_VARIANT')
if selected_variant:variants=[v for v in variants if v[0]==selected_variant]
predrows=[];foldrows=[];searchrows=[];audit=[]
for variant,mask,fc,ncomp,names,residual in variants:
 predrows=[];foldrows=[];searchrows=[];audit=[]
 ix=np.flatnonzero(mask);groups=sites[ix]
 for outer,(a,b) in enumerate(LeaveOneGroupOut().split(ix,y[ix],groups)):
  tr=ix[a];te=ix[b];held=sites[te][0];checkpoint=C/f'{variant}_{held}.json'
  if checkpoint.exists():
   saved=json.load(open(checkpoint));predrows+=saved['pred'];foldrows+=saved['fold'];searchrows+=saved['search'];audit+=saved['audit'];continue
  assert not set(sites[tr])&set(sites[te]);assert not set(ids[tr])&set(ids[te])
  inner=[];splits=[]
  for itr,iva in GroupKFold(n_splits=3).split(tr,y[tr],sites[tr]):
   ti=tr[itr];vi=tr[iva];assert not set(sites[ti])&set(sites[vi]);assert not set(sites[ti])&set(sites[te])
   assert len(np.unique(y[ti]))==2 and len(np.unique(y[vi]))==2
   data,_=prepare(ti,vi,fc,ncomp,residual);inner.append((data,y[ti],y[vi]));splits.append({'train_ids':ids[ti].tolist(),'validation_ids':ids[vi].tolist(),'train_centers':sorted(set(sites[ti])),'validation_centers':sorted(set(sites[vi]))})
  full,explained=prepare(tr,te,fc,ncomp,residual)
  pp=[];ff=[];ss=[];aa=[{'variant':variant,'heldout_center':held,'train_ids':ids[tr].tolist(),'test_ids':ids[te].tolist(),'train_centers':sorted(set(sites[tr])),'inner':splits,'pca_components':ncomp,'pca_explained_variance':explained}]
  for mode in (['connectome'] if residual else ['demographic','connectome','combined']):
   for name in names:
    values=[]
    for params in grids(name):
     aucs=[]
     for data,yt,yv in inner:
      xt,xv=data[mode];m=estimator(name,params).fit(xt,yt);score=m.decision_function(xv) if name=='SVM' else m.predict_proba(xv)[:,1];aucs.append(roc_auc_score(yv,score))
     values.append(float(np.mean(aucs)));ss.append({'variant':variant,'heldout_center':held,'feature_set':mode,'model':name,'params':json.dumps(params),'inner_mean_auc':values[-1]})
    params=grids(name)[int(np.argmax(values))];xt,xv=full[mode];m=estimator(name,params).fit(xt,y[tr]);score=m.decision_function(xv) if name=='SVM' else m.predict_proba(xv)[:,1];pred=m.predict(xv)
    ff.append({'variant':variant,'heldout_center':held,'feature_set':mode,'model':name,**measure(y[te],score,pred),'params':json.dumps(params),'inner_best_auc':max(values),'pca_explained_variance':explained})
    for k,j in enumerate(te):pp.append({'variant':variant,'feature_set':mode,'model':name,'file_id':str(ids[j]),'raw_site':str(rawsites[j]),'center':str(sites[j]),'y_true':int(y[j]),'y_pred':int(pred[k]),'score':float(score[k]),'score_type':'margin' if name=='SVM' else 'probability'})
  checkpoint.write_text(json.dumps({'pred':pp,'fold':ff,'search':ss,'audit':aa}))
  predrows+=pp;foldrows+=ff;searchrows+=ss;audit+=aa
  pd.DataFrame(predrows).to_csv(R/f'predictions_{variant}.csv',index=False);pd.DataFrame(foldrows).to_csv(R/f'site_scores_{variant}.csv',index=False)
  print(variant,held,'ntrain',len(tr),'ntest',len(te),'conditions',len(ff),'PCA variance',round(explained,3),flush=True)
 pd.DataFrame(predrows).to_csv(R/f'predictions_{variant}.csv',index=False);pd.DataFrame(foldrows).to_csv(R/f'site_scores_{variant}.csv',index=False);pd.DataFrame(searchrows).to_csv(R/f'inner_search_{variant}.csv',index=False)
 (R/f'split_audit_{variant}.json').write_text(json.dumps(audit))
print('All planned model evaluations completed',flush=True)
