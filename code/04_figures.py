import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve,confusion_matrix,ConfusionMatrixDisplay
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
s=pd.read_csv(R/'summary_with_CI.csv');p=pd.read_csv(R/'predictions.csv');fold=pd.read_csv(R/'site_scores.csv');c=pd.read_csv(R/'analysis_cohort.csv');flow=json.load(open(R/'cohort_flow.json'));added=pd.read_csv(R/'primary_added_value.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':8,'lines.linewidth':2,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
colors={'LR':'#0072B2','SVM':'#D55E00','RF':'#009E73'};styles={'LR':'-','SVM':'--','RF':':'};modes=['demographic','connectome','combined'];names=['LR','SVM','RF'];rng=np.random.default_rng(2026)
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.pdf'),bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.tiff'),dpi=300,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'});from PIL import Image
 im=Image.open(F/(name+'.tiff')).convert('RGB');im.save(F/(name+'.tiff'),compression='tiff_lzw',dpi=(300,300));plt.close(fig)
fig,ax=plt.subplots(figsize=(7.087,5.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc='#f1f5f9',ec='#334155',lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#334155'})
reasons=flow['sequential_reasons'];ndownload=flow['metadata_download_n'];n=flow['primary_n'];manual=reasons.get('manual_quality_fail',0);timefail=reasons.get('download_or_timeseries_QC_failure',0)
box(.05,.80,.45,.13,f'ABIDE I phenotypic records\nN = {flow["phenotypic_n"]}');box(.61,.80,.34,.13,'Metadata exclusions (n = 161)\nNo derivative filename: 77\nUnavailable / high FD: 84')
arrow((.50,.865),(.61,.865));box(.05,.56,.45,.13,f'Valid file and mean FD 0–0.30 mm\nROI retrieval attempted: N = {ndownload}');arrow((.275,.8),(.275,.69))
box(.61,.52,.34,.21,f'Further exclusions (sequential)\nAny manual QC fail: {manual}\nRetrieval / time-series QC: {timefail}\n111 consistent ROIs required\nAt least 70 finite time points');arrow((.50,.62),(.61,.62))
box(.05,.28,.45,.17,f'Primary cohort N = {n}\nASD {flow["classes"].get("1",0)} / control {flow["classes"].get("2",0)}\n17 institution groups\nNo class-balanced subsampling');arrow((.275,.56),(.275,.45))
box(.61,.25,.34,.20,f'Observed edge missingness\n{flow["missing_edge_subjects"]} participants\nTraining-only median imputation\nSensitivity: complete cases\nand removal of ROI #3455');arrow((.5,.365),(.61,.365))
box(.05,.04,.90,.12,'Outer leave-one-center-out evaluation\nInner 3-fold GroupKFold by center; every participant held out once');arrow((.275,.28),(.275,.16));save(fig,'Figure_1_participant_flow')
fig,axes=plt.subplots(1,3,figsize=(7.087,2.8));grid=np.linspace(0,1,201)
for ax,mode,panel in zip(axes,modes,['A','B','C']):
 for name in names:
  sub=p[(p.variant=='primary')&(p.feature_set==mode)&(p.model==name)];curves=[]
  for _,g in sub.groupby('center'):
   if g.y_true.nunique()<2:continue
   x,y,_=roc_curve(g.y_true,g.score);curves.append(np.interp(grid,x,y))
  mean=np.mean(curves,axis=0);mean[0]=0;mean[-1]=1
  point=s[(s.variant=='primary')&(s.feature_set==mode)&(s.model==name)].iloc[0]
  ax.plot(grid,mean,styles[name],color=colors[name],label=f'{name} {point.macro_roc_auc:.3f}')
 ax.plot([0,1],[0,1],color='#777777',lw=1,ls='--');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'({panel}) {mode}',xlim=(0,1),ylim=(0,1));ax.legend(loc='lower right');ax.grid(alpha=.2)
fig.tight_layout(pad=.7);save(fig,'Figure_2_center_mean_ROC')
fig,axes=plt.subplots(1,2,figsize=(7.087,3.6),gridspec_kw={'width_ratios':[1.5,1]})
for j,mode in enumerate(modes):
 for k,name in enumerate(names):
  r=s[(s.variant=='primary')&(s.feature_set==mode)&(s.model==name)].iloc[0];yy=j*4+k
  axes[0].errorbar(r.macro_roc_auc,yy,xerr=np.array([[r.macro_roc_auc-r.roc_auc_ci_low],[r.roc_auc_ci_high-r.macro_roc_auc]]),fmt=['o','s','^'][k],color=colors[name],capsize=2,lw=1.5)
axes[0].set_yticks([j*4+k for j in range(3) for k in range(3)],[mode+' / '+name for mode in modes for name in names]);axes[0].invert_yaxis();axes[0].axvline(.5,color='#777',ls='--',lw=1);axes[0].set(xlabel='Equal-center mean AUC and 95% CI',xlim=(.2,.85),title='(A) Model and feature comparison')
for k,name in enumerate(names):
 r=added[added.model==name].iloc[0];axes[1].errorbar(r.delta_macro_auc,k,xerr=[[r.delta_macro_auc-r.ci_low],[r.ci_high-r.delta_macro_auc]],fmt='o',color=colors[name],capsize=3,lw=1.5)
axes[1].axvline(0,color='#777',ls='--',lw=1);axes[1].set_yticks(range(3),names);axes[1].invert_yaxis();axes[1].set(xlabel='Combined minus demographic AUC',title='(B) Added value of connectivity');fig.tight_layout();save(fig,'Figure_3_model_and_added_value')
fig,axes=plt.subplots(1,3,figsize=(7.087,5.5),sharey=True)
centers=sorted(c.center.unique())
for ax,mode,panel in zip(axes,modes,['A','B','C']):
 for k,name in enumerate(names):
  g=fold[(fold.variant=='primary')&(fold.feature_set==mode)&(fold.model==name)].set_index('heldout_center').reindex(centers)
  ax.plot(g.roc_auc,np.arange(len(centers))+(k-1)*.15,marker=['o','s','^'][k],markersize=4,lw=0,linestyle='None',color=colors[name],label=name)
 ax.axvline(.5,color='#777',ls='--',lw=1);ax.set(xlim=(0,1),xlabel='Held-out center AUC',title=f'({panel}) {mode}');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(centers)),centers);axes[0].invert_yaxis();axes[-1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3);fig.tight_layout();save(fig,'Figure_4_per_center_performance')
fig,axes=plt.subplots(3,3,figsize=(7.087,7.3))
for i,mode in enumerate(modes):
 for j,name in enumerate(names):
  g=p[(p.variant=='primary')&(p.feature_set==mode)&(p.model==name)];ConfusionMatrixDisplay(confusion_matrix(g.y_true,g.y_pred,labels=[0,1]),display_labels=['Control','ASD']).plot(ax=axes[i,j],colorbar=False,cmap='Blues',values_format='d');axes[i,j].set_title(mode+' / '+name)
fig.tight_layout();save(fig,'Figure_S1_confusion_matrices')
fig,axes=plt.subplots(2,1,figsize=(7.087,6.0),sharex=True)
for ax,var in zip(axes,['AGE_AT_SCAN','func_mean_fd']):
 for dx,offset,color in [(1,-.16,'#D55E00'),(2,.16,'#0072B2')]:
  vals=[c.loc[(c.center==center)&(c.DX_GROUP==dx),var].dropna().to_numpy() for center in centers]
  bp=ax.boxplot(vals,positions=np.arange(len(centers))+offset,widths=.27,patch_artist=True,showfliers=False,manage_ticks=False)
  for patch in bp['boxes']:patch.set_facecolor(color);patch.set_alpha(.6)
 ax.set_ylabel('Age (years)' if var=='AGE_AT_SCAN' else 'Mean FD (mm)');ax.grid(axis='y',alpha=.2)
axes[1].set_xticks(range(len(centers)),centers,rotation=60,ha='right');axes[0].text(.02,.95,'Orange: ASD     Blue: Control',transform=axes[0].transAxes,va='top');fig.tight_layout();save(fig,'Figure_S2_age_and_motion')
fig,ax=plt.subplots(figsize=(7.087,3.6))
for dx,offset,color in [(1,-.17,'#D55E00'),(2,.17,'#0072B2')]:
 rates=[100*c.loc[(c.center==center)&(c.DX_GROUP==dx),'has_missing_edges'].mean() for center in centers];ax.bar(np.arange(len(centers))+offset,rates,width=.32,color=color,label='ASD' if dx==1 else 'Control')
ax.set_xticks(range(len(centers)),centers,rotation=60,ha='right');ax.set(ylabel='Participants with missing edges (%)');ax.legend();fig.tight_layout();save(fig,'Figure_S3_missingness_by_center')
fig,ax=plt.subplots(figsize=(7.087,4.0));variants=['primary','complete_case','drop_roi3455','low_motion','pca8','strict_manual','residualized']
for i,mode in enumerate(modes):
 rr=s[(s.model=='LR')&(s.feature_set==mode)].set_index('variant').reindex(variants);ax.plot(np.arange(len(variants)),rr.macro_roc_auc,marker=['o','s','^'][i],label=mode,lw=1.5)
ax.set_xticks(range(len(variants)),[v.replace('_',' ') for v in variants],rotation=25,ha='right');ax.set(ylabel='Equal-center mean ROC-AUC',ylim=(.2,.85));ax.axhline(.5,color='#777',ls='--',lw=1);ax.legend();ax.grid(alpha=.2);fig.tight_layout();save(fig,'Figure_S4_sensitivity_LR')
print('Publication figure files created:',len(list(F.glob('*'))))
