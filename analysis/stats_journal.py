"""Revision statistics computed ONLY from saved artefacts in code/results/journal.
No training is run. Usage: python3 revision_stats.py <results_dir> <out_json>"""
import json, sys, numpy as np
from scipy import stats
R, OUT = sys.argv[1], sys.argv[2]
J = lambda f: json.load(open(f"{R}/{f}"))
out = {}
def ms(a): a=np.asarray(a,float); return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), n=int(len(a)))
def paired(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); d=a-b
    t=stats.ttest_rel(a,b)
    try: w=stats.wilcoxon(a,b,zero_method='wilcox',alternative='two-sided',mode='exact' if np.all(d!=0) else 'auto')
    except ValueError: w=None
    se=d.std(ddof=1)/np.sqrt(len(d)); h=stats.t.ppf(.975,len(d)-1)*se
    return dict(diff=float(d.mean()), ci=[float(d.mean()-h),float(d.mean()+h)], t_p=float(t.pvalue),
                wilcoxon_W=(float(w.statistic) if w else None), wilcoxon_p=(float(w.pvalue) if w else None),
                n_nonzero=int((d!=0).sum()), a_ahead=int((d>0).sum()), n=int(len(d)))
def fisher(c1,n1,c2,n2):
    return float(stats.fisher_exact([[c1,n1-c1],[c2,n2-c2]])[1])
COL=0.10

# 1 stranded stock
pc=J('policy_comparison_R2f-bind.json'); ps=pc['per_seed']
out['stranded']={p:dict(stranded=ms([x['stranded'] for x in ps[p]]), energy=ms([x['energy_j'] for x in ps[p]]),
      updates=ms([x['updates'] for x in ps[p]]), terminal=ms([x['A_final'] for x in ps[p]]),
      collapsed=int(sum(x['A_final']<COL for x in ps[p])))
      for p in ['FedAvg-full','Threshold-0.3','Threshold-0.0','DRR','Threshold-0.3+deficit-order','Threshold-0.3+minsel','FedAvg-full+minsel','Threshold-0.3+minwidth','DRR+minwidth','FedAvg-full+minwidth']}
g=lambda p,k='A_final':[x[k] for x in sorted(ps[p],key=lambda x:x['seed'])]
out['minsel_tests']={'Threshold+minsel vs Threshold':paired(g('Threshold-0.3+minsel'),g('Threshold-0.3')),
                     'FedAvg+minsel vs FedAvg':paired(g('FedAvg-full+minsel'),g('FedAvg-full')),
                     'DRR+minwidth vs DRR':paired(g('DRR+minwidth'),g('DRR'))}
# 2 order-only
oo=J('order_only_R2f-bind.json')
T={a:[r['terminal'] for r in sorted(oo['rows'],key=lambda r:r['seed']) if r['arm']==a] for a in ['published','deficit','percost','random']}
out['order_only']={a:dict(**ms(T[a]),collapsed=int(sum(v<COL for v in T[a])),values=T[a]) for a in T}
out['order_only_tests']={f'{a} vs published':paired(T[a],T['published']) for a in ['deficit','percost','random']}
out['order_only_tests']['deficit vs percost']=paired(T['deficit'],T['percost'])
c=lambda a:int(sum(v<COL for v in T[a]))
out['order_only_fisher']={'deficit(0/10) vs published(5/10)':fisher(c('deficit'),10,c('published'),10),
                          'random vs published':fisher(c('random'),10,c('published'),10),
                          'deficit vs random':fisher(c('deficit'),10,c('random'),10)}
# 3 two-level accuracy
tl=J('two_level_accuracy_R2f-bind.json'); arms=sorted(set(r['arm'] for r in tl['rows'])); out['twolevel_arms']=arms
def tv(arm,k,key='terminal'): return [r[key] for r in sorted(tl['rows'],key=lambda r:r['seed']) if r['arm']==arm and r['kappa']==k]
out['twolevel']={}
for k in tl['kappas']:
    for a in arms:
        v=tv(a,k); out['twolevel'][f'k{k}|{a}']=dict(**ms(v),collapsed=int(sum(x<COL for x in v)),
            depleted=float(np.mean(tv(a,k,'depleted'))),energy=float(np.mean(tv(a,k,'energy_j'))),parts=float(np.mean(tv(a,k,'participations'))))
out['twolevel_tests']={}
for k in tl['kappas']:
    cand=[a for a in arms]
    out['twolevel_tests'][f'k{k}']={f'{a} vs {b}':paired(tv(a,k),tv(b,k)) for a in cand for b in cand if a<b}
# pooled collapse counts: deficit order vs published order across the two studies
# 4 endpoint (work quantum)
wq=J('work_quantum_R2f-bind.json'); pols=sorted(set(r['policy'] for r in wq['rows'])); out['wq_policies']=pols
out['endpoint']={}
for p in pols:
  for w in wq['quanta']:
    rr=[r for r in sorted(wq['rows'],key=lambda r:r['seed']) if r['policy']==p and r['w']==w]
    if not rr: continue
    a=[r['final_macro_f1'] for r in rr]; b=[r['mean_last5'] for r in rr]
    out['endpoint'][f'{p}|w{w}']=dict(terminal=ms(a),tail5=ms(b),test=paired(a,b),final_width=[r['final_width'] for r in rr],
        min_width=[r['min_width'] for r in rr],collapsed=int(sum(x<COL for x in a)),terminal_values=a,
        energy=float(np.mean([r['energy_j'] for r in rr])))
# 5 family selection
fs=J('policy_family_selection_balanced_R2f-bind.json'); fams=list(fs['families'])
out['families']={f:dict(test=ms([r[f'test_{f}'] for r in fs['rows']]),J=ms([r[f'J_{f}'] for r in fs['rows']]),
                        tail=ms([r[f'tail_{f}'] for r in fs['rows']]),chosen=[r[f'chosen_{f}'] for r in fs['rows']]) for f in fams}
out['families_tests']={f'{f} vs DRR':paired([r[f'test_{f}'] for r in fs['rows']],[r['test_DRR'] for r in fs['rows']]) for f in fams if f!='DRR'}
out['families_row_keys']=[k for k in fs['rows'][0] if not k.startswith(('grid_','val_grid'))]
# energy-normalised reading: F1 per joule, and Spearman between J and score difference
for f in fams:
    if f=='DRR': continue
    dJ=np.array([r['J_DRR']-r[f'J_{f}'] for r in fs['rows']]); dF=np.array([r['test_DRR']-r[f'test_{f}'] for r in fs['rows']])
    sp=stats.spearmanr(dJ,dF); out['families_tests'][f'{f} vs DRR']['dJ_mean']=float(dJ.mean()); out['families_tests'][f'{f} vs DRR']['spearman_dJ_dF']=[float(sp.statistic),float(sp.pvalue)]
# 6 SBC: tier decoupling
td=J('order_tier_decoupling_R2f-bind.json'); out['tier_decoupling']={}
keys=sorted(set((r['variant'],r['epochs'],r['tau']) for r in td))
for v,e,t in keys:
    a=[r['terminal'] for r in sorted(td,key=lambda r:r['seed']) if (r['variant'],r['epochs'],r['tau'],r['order'])==(v,e,t,'deficit')]
    b=[r['terminal'] for r in sorted(td,key=lambda r:r['seed']) if (r['variant'],r['epochs'],r['tau'],r['order'])==(v,e,t,'published')]
    out['tier_decoupling'][f'{v}|w{e}|tau{t}']=dict(deficit=ms(a),published=ms(b),test=paired(a,b),
        collapsed_deficit=int(sum(x<COL for x in a)),collapsed_published=int(sum(x<COL for x in b)),
        fisher=fisher(int(sum(x<COL for x in a)),len(a),int(sum(x<COL for x in b)),len(b)))
rm=[r for r in td if r['variant']=='sbc_removed']
out['sbc_removed_meta']=dict(n_dropped=sorted(set(r['n_dropped'] for r in rm)),classes_present=sorted(set(tuple(r['classes_present']) for r in rm)))
# 7 clean vs full federation
cf=J('twolevel_preservation_clean_vs_full_federation.json'); out['clean_vs_full']={}
for reg in ['R2f-bind','R2f-slack']:
  for clean in [False,True]:
    for arm in sorted(set(r['arm'] for r in cf)):
      rr=[r for r in sorted(cf,key=lambda r:r['seed']) if r['regime']==reg and r['clean']==clean and r['arm']==arm]
      v=[r['terminal'] for r in rr]
      out['clean_vs_full'][f'{reg}|{"clean" if clean else "full"}|{arm}']=dict(terminal=ms(v),tail5=ms([r['tail5'] for r in rr]),best=ms([r['best'] for r in rr]),
         collapsed=int(sum(x<COL for x in v)),exhausted=float(np.mean([r['exhausted'] for r in rr])),n_part=float(np.mean([r['n_participating'] for r in rr])),
         stranded=float(np.mean([r['stranded'] for r in rr])),energy=float(np.mean([r['energy'] for r in rr])),final_width=[r['final_width'] for r in rr],values=v)
def cv(reg,clean,arm): return out['clean_vs_full'][f'{reg}|{clean}|{arm}']['values']
out['clean_tests']={}
for reg in ['R2f-bind','R2f-slack']:
  for clean in ['full','clean']:
    for a,b in [('DRR','ResFed'),('ResFed+paced','ResFed'),('ResFed+guard','ResFed'),('FedAvg','ResFed'),('DRR','FedAvg')]:
      out['clean_tests'][f'{reg}|{clean}|{a} vs {b}']=paired(cv(reg,clean,a),cv(reg,clean,b))
# 8 tau sweep: gate_order (full federation) and threshold_central_reference (clean)
go=J('gate_order.json'); out['gate_order_summary']=go['summary']; out['gate_order']={}
for reg in sorted(set(r['regime'] for r in go['rows'])):
  for e in go['epochs']:
    for t in go['taus']:
      d={}
      for o in sorted(set(r['order'] for r in go['rows'])):
        v=[r['terminal'] for r in sorted(go['rows'],key=lambda r:r['seed']) if (r['regime'],r['epochs'],r['tau'],r['order'])==(reg,e,t,o)]
        en=[r['energy_j'] for r in go['rows'] if (r['regime'],r['epochs'],r['tau'],r['order'])==(reg,e,t,o)]
        up=[r['updates'] for r in go['rows'] if (r['regime'],r['epochs'],r['tau'],r['order'])==(reg,e,t,o)]
        d[o]=dict(**ms(v),collapsed=int(sum(x<COL for x in v)),energy=float(np.mean(en)),updates=float(np.mean(up)),values=v)
      os_=list(d); d['test']=paired(d[os_[0]]['values'],d[os_[1]]['values']); d['test_label']=f'{os_[0]} vs {os_[1]}'
      out['gate_order'][f'{reg}|w{e}|tau{t}']=d
tc=J('threshold_central_reference.json'); out['tau_clean']={}
for reg in ['R2f-bind','R2f-slack']:
  for t in [0.2,0.5]:
    rr=[r for r in tc if r['regime']==reg and r['tau']==t]
    out['tau_clean'][f'{reg}|tau{t}']=dict(terminal=ms([r['terminal'] for r in rr]),stranded=float(np.mean([r['stranded'] for r in rr])),
        exhausted=float(np.mean([r['exhausted'] for r in rr])),energy=float(np.mean([r['energy'] for r in rr])),updates=float(np.mean([r['updates'] for r in rr])),
        collapsed=int(sum(r['terminal']<COL for r in rr)),n_part=float(np.mean([r['n_participating'] for r in rr])))
rr=[r for r in tc if r['regime']=='central']; out['central_clean']=dict(terminal=ms([r['terminal'] for r in rr]),best=ms([r['best'] for r in rr]))
# 9 misspecification summaries
mm=J('model_misspecification_R2f-bind.json'); out['misspec']=mm['summary']; out['misspec_meta']={k:mm[k] for k in ['tau','delta','noise','kappas']}
json.dump(out,open(OUT,'w'),indent=1)
print('written',OUT)
