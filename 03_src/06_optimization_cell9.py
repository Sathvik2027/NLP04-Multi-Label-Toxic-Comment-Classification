# Notebook 06, Cell 9: finalize validation analysis without training
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

registry9 = pd.read_csv(REGISTRY_PATH, dtype=str, keep_default_na=False)
assert len(registry9) == 19 and registry9['run_id'].is_unique
assert registry9['status'].eq('Completed').all()
registry9_hash = calculate_sha256(REGISTRY_PATH)
decision9_hash = calculate_sha256(DECISION_LOG_PATH)
assert calculate_sha256(VALIDATION_PATH) == EXPECTED_VALIDATION_HASH
assert calculate_sha256(SRC_DIR/'evaluation_utils.py') == evaluation_hash

val9 = pd.read_parquet(VALIDATION_PATH, columns=['id','group_id',*LABELS])
val9['id'] = val9['id'].astype(str)
assert len(val9) == 31911 and val9['id'].is_unique
assert val9['group_id'].notna().all()
y9 = val9[LABELS].to_numpy(dtype=np.int8)
assert np.isin(y9,[0,1]).all()

# Resolve and normalize historical and current validation artifacts in memory.
import pyarrow.parquet as pq


def owned_path9(path, rid, root):
    parts = [part.lower() for part in path.relative_to(PROJECT_ROOT).parts]
    if any(rid.lower() in part for part in parts):
        return True
    # A registry results directory used by only one run establishes ownership.
    owners = registry9.loc[registry9['results_path'].map(
        lambda value: (PROJECT_ROOT / value).resolve() == root.resolve()), 'run_id']
    return len(owners) == 1 and owners.iloc[0] == rid


def prediction_mapping9(columns):
    columns = set(columns)
    if 'id' not in columns:
        return None
    mapping = {}
    for label in LABELS:
        for role in ('true', 'score', 'pred_default', 'pred_optimized'):
            names = [f'{role}_{label}', f'{label}_{role}']
            present = [name for name in names if name in columns]
            if len(present) > 1:
                raise AssertionError(f'Ambiguous columns: {present}')
            if present:
                mapping[present[0]] = f'{role}_{label}'
            elif role in ('true', 'score'):
                return None
    return mapping


def prediction9(path, rid, root, declared=False):
    if path.suffix.lower() == '.parquet':
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows != len(val9):
            return None
        mapping = prediction_mapping9(pf.schema_arrow.names)
        if mapping is None:
            return None
        columns = pf.schema_arrow.names
        selected = ['id', *mapping, *[c for c in ('run_id','split') if c in columns]]
        p = pd.read_parquet(path, columns=selected)
    elif path.suffix.lower() == '.csv':
        columns = pd.read_csv(path, nrows=0).columns
        mapping = prediction_mapping9(columns)
        if mapping is None:
            return None
        selected = ['id', *mapping, *[c for c in ('run_id','split') if c in columns]]
        p = pd.read_csv(path, usecols=selected, dtype={'id':str}, float_precision='round_trip')
        if len(p) != len(val9):
            return None
    else:
        return None
    if 'run_id' in p:
        if not p['run_id'].eq(rid).all():
            return None
    elif not (declared or owned_path9(path,rid,root)):
        return None
    if 'split' in p:
        if not p['split'].astype(str).str.lower().eq('validation').all():
            return None
    elif not declared and 'validation' not in str(path.relative_to(root)).lower():
        return None
    p = p.rename(columns=mapping)
    p['id'] = p['id'].astype(str)
    assert p['id'].is_unique and set(p['id']) == set(val9['id']), f'{rid}: ID mismatch in {path}'
    p = p.set_index('id').loc[val9['id']].reset_index()
    np.testing.assert_array_equal(p[[f'true_{l}' for l in LABELS]].to_numpy(),y9)
    assert np.isfinite(p[[f'score_{l}' for l in LABELS]].to_numpy()).all(), path
    p['run_id'] = rid
    p['split'] = 'validation'
    return p


def threshold9(path, rid, root, config, declared=False):
    obj = read_json(path)
    if not isinstance(obj,dict):
        return None
    if obj.get('run_id') not in (None,rid):
        return None
    if obj.get('run_id') is None and not (declared or owned_path9(path,rid,root)):
        return None
    keys = [k for k in ('optimized_thresholds','thresholds') if isinstance(obj.get(k),dict)]
    if not keys:
        return None
    thresholds = obj[keys[0]]
    if set(thresholds) != set(LABELS):
        return None
    assert all(obj[k] == thresholds for k in keys), path
    labels = obj.get('labels',config.get('labels'))
    assert labels == LABELS, path
    assert obj.get('selection_partition',obj.get('threshold_selection_partition')) == 'validation', path
    assert obj.get('official_test_used',config.get('official_test_used')) is False, path
    operator = obj.get('comparison_operator',obj.get('threshold_comparison_operator',config.get('threshold_comparison_operator')))
    assert operator in ('>=','greater_than_or_equal','greater_than_or_equal_to'), f'{rid}: unknown comparison operator {operator}'
    score_type = obj.get('score_type',config.get('score_type'))
    assert isinstance(score_type,str) and score_type, path
    default = obj.get('default_threshold',config.get('default_threshold'))
    assert default is not None, f'{rid}: default threshold not recorded'
    default = float(default)
    thresholds = {l:float(thresholds[l]) for l in LABELS}
    assert np.isfinite([default,*thresholds.values()]).all(), path
    return {**obj,'run_id':rid,'labels':labels,'optimized_thresholds':thresholds,
            'comparison_operator':'>=','selection_partition':'validation','official_test_used':False,
            'score_type':score_type,'default_threshold':default}


def resolve_run_artifacts9(row):
    rid = row['run_id']
    root = PROJECT_ROOT / row['results_path']
    assert root.is_dir(), root
    config_path = PROJECT_ROOT / row['config_path']
    config = read_json(config_path)
    assert config.get('run_id') == rid, config_path
    assert config.get('official_test_used') is False, config_path
    metadata = [config]
    for p in sorted(root.rglob('*summary*.json')):
        obj = read_json(p)
        if isinstance(obj,dict) and obj.get('run_id') == rid:
            metadata.append(obj)
    resolved = []
    for kind,keys in [('predictions',('predictions_path','prediction_path','validation_predictions_path','validation_scores_path')),
                      ('thresholds',('threshold_path','thresholds_path','threshold_config_path'))]:
        declared = set()
        for meta in metadata:
            for key in keys:
                value = meta.get(key)
                if not isinstance(value,str) or not value:
                    continue
                candidates = [PROJECT_ROOT/value,root/value,config_path.parent/value]
                paths = {p.resolve() for p in candidates if p.is_file()}
                assert len(paths)==1, f'{rid}: cannot uniquely resolve recorded {key}: {value}'
                path = paths.pop()
                assert path.is_relative_to(PROJECT_ROOT.resolve()), path
                expected = meta.get(key.replace('_path','_sha256'))
                if expected:
                    assert calculate_sha256(path)==expected, path
                declared.add(path)
        candidates = sorted(declared) if declared else sorted(p for p in root.rglob('*') if p.is_file() and
            ((kind=='predictions' and p.suffix.lower() in ('.parquet','.csv') and
              ('pred' in p.name.lower() or 'score' in p.name.lower() or 'pred' in p.parent.name.lower()) and 'test' not in p.name.lower()) or
             (kind=='thresholds' and p.suffix.lower()=='.json' and 'threshold' in p.name.lower())))
        matches = []
        for path in candidates:
            if kind=='predictions':
                obj = prediction9(path,rid,root,path in declared)
            else:
                obj = threshold9(path,rid,root,config,path in declared)
            if obj is not None:
                matches.append((path,obj))
            elif path in declared:
                raise AssertionError(f'{rid}: invalid declared {kind}: {path}')
        assert matches, f'{rid}: no matching {kind}; candidates: {[str(p) for p in candidates]}'
        if kind=='predictions':
            # Score-only and decision files may have different bytes but identical evidence.
            matches.sort(key=lambda pair:(-len(pair[1].columns),str(pair[0])))
            canonical = ['id',*[f'true_{l}' for l in LABELS],*[f'score_{l}' for l in LABELS]]
            first = matches[0][1]
            for path,obj in matches[1:]:
                pd.testing.assert_frame_equal(first[canonical],obj[canonical],check_dtype=False,check_exact=True)
        else:
            first = matches[0][1]
            for path,obj in matches[1:]:
                for key in ('optimized_thresholds','default_threshold','score_type','comparison_operator'):
                    assert first[key]==obj[key], f'{rid}: conflicting thresholds: {path}'
        resolved.append(matches[0])
    return resolved[0], resolved[1]


# Resolve all runs before computing metrics or writing any new outputs.
resolved9 = {}
issues9 = []
for row in registry9.to_dict('records'):
    rid = row['run_id']
    try:
        resolved9[rid] = resolve_run_artifacts9(row)
        print('Resolved:',rid,flush=True)
    except Exception as exc:
        issues9.append(f'{rid}: {type(exc).__name__}: {exc}')
if issues9:
    print('\nArtifact preflight issues:')
    for issue in issues9:
        print(issue)
    raise RuntimeError('Resolve the listed issues before proceeding. No new outputs written.')

ranking9, thresholds9, sources9 = [], [], []
scores9, per_label9 = {}, {}
for row in registry9.to_dict('records'):
    rid = row['run_id']
    root = PROJECT_ROOT/row['results_path']
    (pp,p), (tp,cfg) = resolved9[rid]
    assert cfg['run_id']==rid and cfg['labels']==LABELS
    assert cfg['selection_partition']=='validation'
    assert not cfg['official_test_used']
    assert cfg['comparison_operator']=='>='
    cutoff = np.array([cfg['optimized_thresholds'][label] for label in LABELS],dtype=np.float64)
    default = float(cfg['default_threshold'])
    assert np.isfinite(cutoff).all() and np.isfinite(default)
    assert p is not None, pp
    assert p['run_id'].eq(rid).all() and p['split'].eq('validation').all()
    p['id'] = p['id'].astype(str)
    assert p['id'].is_unique and len(p)==len(val9)
    assert set(p['id'])==set(val9['id'])
    p = p.set_index('id').loc[val9['id']]
    np.testing.assert_array_equal(p[[f'true_{l}' for l in LABELS]].to_numpy(), y9)
    score = p[[f'score_{l}' for l in LABELS]].to_numpy()
    assert np.isfinite(score).all()
    for role, threshold in [('pred_default',default),('pred_optimized',cutoff)]:
        columns = [f'{role}_{l}' for l in LABELS]
        if set(columns).issubset(p.columns):
            np.testing.assert_array_equal(p[columns].to_numpy(), apply_thresholds(score,threshold))
    d, _ = evaluate_multilabel(y9,score,LABELS,thresholds=default)
    o, labels = evaluate_multilabel(y9,score,LABELS,thresholds=cutoff)
    assert abs(o['Macro AP']-float(row['validation_macro_ap'])) < 1e-6, rid
    assert abs(o['Macro F1']-float(row['validation_macro_f1'])) < 1e-6, rid
    assert o['Macro F1'] >= d['Macro F1']-1e-12, rid
    ranking9.append({'Run ID':rid,'Model':row['model'],'Loss':row['loss_function'],
        'Sampling':row['sampling_strategy'],'Training Samples':int(float(row['training_samples'])),
        'Macro AP':o['Macro AP'],'Macro ROC-AUC':o['Macro ROC-AUC'],
        'Default Macro F1':d['Macro F1'],'Optimized Macro F1':o['Macro F1'],
        'Default Micro F1':d['Micro F1'],'Optimized Micro F1':o['Micro F1'],
        'Training Time Seconds':float(row['training_time_seconds'])})
    scores9[rid]=score
    per_label9[rid]=labels
    for i,label in enumerate(LABELS):
        thresholds9.append({'Run ID':rid,'Label':label,'Score Type':cfg['score_type'],
            'Default Threshold':default,'Optimized Threshold':cutoff[i],
            'Validation F1':float(labels.loc[labels['Label'].eq(label),'F1'].iloc[0])})
    sources9.append({'run_id':rid,'predictions_path':str(pp.relative_to(PROJECT_ROOT)),
        'predictions_sha256':calculate_sha256(pp),'threshold_path':str(tp.relative_to(PROJECT_ROOT)),
        'threshold_sha256':calculate_sha256(tp)})
    print('Reconstructed validation metrics:',rid,flush=True)

ranking9 = pd.DataFrame(ranking9).sort_values(['Macro AP','Optimized Macro F1','Run ID'],ascending=[False,False,True]).reset_index(drop=True)
ranking9.insert(0,'Rank',np.arange(1,len(ranking9)+1))
thresholds9=pd.DataFrame(thresholds9)
selected9=ranking9.iloc[0]['Run ID']
assert selected9=='S02_ROBERTA_FOCAL_FULL_S42'
print('\nValidation-only model ranking')
display(ranking9.round(6))

# Compare completed full-data experiments with their matched references.
scaling9=[]
for base,full in [('LR_COMBINED_BALANCED_S42','S01_LR_COMBINED_BALANCED_FULL_S42'),('T06_FOCAL_S42','S02_ROBERTA_FOCAL_FULL_S42')]:
    a=ranking9.set_index('Run ID').loc[base]; b=ranking9.set_index('Run ID').loc[full]
    for metric in ['Training Samples','Macro AP','Macro ROC-AUC','Default Macro F1','Optimized Macro F1','Optimized Micro F1','Training Time Seconds']:
        scaling9.append({'Reference Run':base,'Full Run':full,'Metric':metric,'Representative':float(a[metric]),'Full':float(b[metric]),'Change':float(b[metric]-a[metric])})
scaling9=pd.DataFrame(scaling9)

# Extend the existing stability study to the new leading model only.
# This is a separate fixed-model group bootstrap, not a paired model comparison.
old_stability_path=RUN_CONFIGS_DIR/'threshold_stability_analysis.json'
old_stability=read_json(old_stability_path)
assert not old_stability['official_test_used']
assert old_stability['bootstrap_replicates']==200
assert not old_stability['original_thresholds_changed']
score=scores9[selected9]
cutoffs=thresholds9.loc[thresholds9['Run ID'].eq(selected9)].set_index('Label').loc[LABELS,'Optimized Threshold'].to_numpy()
groups, unique_groups=pd.factorize(val9['group_id'],sort=True)
assert len(unique_groups)==31857
B=200
bootstrap_seed=int(old_stability['bootstrap_seed'])
rng=np.random.default_rng(bootstrap_seed)

# Pre-sort scores; weighted cumulative counts reproduce resampled PR breakpoints.
sorted_info=[]
for j in range(len(LABELS)):
    order=np.argsort(-score[:,j],kind='stable')
    s=score[order,j]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    sorted_info.append((order,s[ends],ends,y9[order,j].astype(np.float64)))

def weighted_f1_search(info, weights):
    order, candidates, ends, target=info
    w=weights[order]
    total_pos=np.dot(w,target)
    if total_pos<=0 or total_pos>=w.sum():
        return None
    tp=np.cumsum(w*target)[ends]
    predicted=np.cumsum(w)[ends]
    mass=np.diff(np.r_[0,predicted])
    f1=2*tp/(total_pos+predicted)
    keep=mass>0
    maximum=np.max(f1[keep])
    best=np.flatnonzero(keep & np.isclose(f1,maximum,rtol=0,atol=1e-12))[0]
    return float(candidates[best]),float(f1[best])

original_search=[]
for j,info in enumerate(sorted_info):
    result=weighted_f1_search(info,np.ones(len(y9),dtype=np.int64))
    np.testing.assert_allclose(result[0],cutoffs[j],atol=1e-12,rtol=0)
    expected=per_label9[selected9].set_index('Label').loc[LABELS[j],'F1']
    np.testing.assert_allclose(result[1],expected,atol=1e-12,rtol=0)
    original_search.append(result)
print('\nThreshold search checked against all six original cutoffs.')

boot9=[]
for b in range(B):
    count=np.bincount(rng.integers(0,len(unique_groups),size=len(unique_groups)),minlength=len(unique_groups))
    weights=count[groups]
    for j,info in enumerate(sorted_info):
        result=weighted_f1_search(info,weights)
        boot9.append({'Run ID':selected9,'Replicate':b+1,'Label':LABELS[j],
            'Optimal Threshold':result[0] if result else np.nan,'Optimized F1':result[1] if result else np.nan})
boot9=pd.DataFrame(boot9)
stability9=[]
for j,label in enumerate(LABELS):
    sub=boot9.loc[boot9['Label'].eq(label),'Optimal Threshold'].dropna()
    assert len(sub)>0
    lo,med,hi=np.percentile(sub,[2.5,50,97.5])
    stability9.append({'Run ID':selected9,'Label':label,'Original Threshold':cutoffs[j],
        'Bootstrap Median':med,'Lower 2.5%':lo,'Upper 97.5%':hi,'Interval Width':hi-lo,
        'Valid Replicates':len(sub),'Total Replicates':B})
stability9=pd.DataFrame(stability9)
print('\nS02 threshold stability (original thresholds retained)')
display(stability9.round(6))

# Save exact F1 at validation score breakpoints for the selected model.
curve9=[]
for j,label in enumerate(LABELS):
    precision,recall,t=precision_recall_curve(y9[:,j],score[:,j])
    f1=np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],out=np.zeros(len(t)),where=(precision[:-1]+recall[:-1])>0)
    curve9.append(pd.DataFrame({'Run ID':selected9,'Label':label,'Threshold':t,'Precision':precision[:-1],'Recall':recall[:-1],'F1':f1}))
curve9=pd.concat(curve9,ignore_index=True)

# Save only the new consolidated evidence; preserve earlier run artifacts.
TABLES_DIR.mkdir(parents=True,exist_ok=True)
FIGURES_DIR.mkdir(parents=True,exist_ok=True)
new_tables={'T01_model_selection.csv':ranking9,'T02_scaling_robustness.csv':scaling9,
    'T03_threshold_search.csv':curve9,'T04_selected_thresholds.csv':thresholds9,
    's02_threshold_bootstrap_replicates.csv':boot9,'s02_threshold_stability.csv':stability9}
for name,frame in new_tables.items():
    save_csv_atomic(TABLES_DIR/name,frame)

fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
for j,ax in enumerate(axes.flat):
    c=curve9[curve9['Label'].eq(LABELS[j])]
    ax.plot(c['Threshold'],c['F1'],lw=1.3)
    ax.axvline(cutoffs[j],ls='--',label='Selected cutoff')
    ax.set_title(LABELS[j]); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Score threshold'); ax.set_ylabel('Validation F1')
axes.flat[0].legend(fontsize=8)
fig.suptitle('S02: validation threshold optimization')
fig.tight_layout(); fig.savefig(FIGURES_DIR/'F03_F1_threshold_panels.png',dpi=300,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(11,8))
ordered=ranking9.iloc[::-1]
positions=np.arange(len(ordered))
ax.barh(positions-.19,ordered['Default Macro F1'],height=.36,label='Default')
ax.barh(positions+.19,ordered['Optimized Macro F1'],height=.36,label='Per-label tuned')
ax.set_yticks(positions,ordered['Run ID'],fontsize=8); ax.set_xlim(0,1)
ax.set_xlabel('Validation Macro F1'); ax.legend(); fig.tight_layout()
fig.savefig(FIGURES_DIR/'F04_default_vs_tuned_validation.png',dpi=300,bbox_inches='tight'); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4))
pairs=[('LR_COMBINED_BALANCED_S42','S01_LR_COMBINED_BALANCED_FULL_S42'),('T06_FOCAL_S42',selected9)]
idx=ranking9.set_index('Run ID'); x=np.arange(2)
for ax,metric in zip(axes,['Macro AP','Training Time Seconds']):
    ax.bar(x-.18,[idx.loc[a,metric] for a,b in pairs],.36,label='Representative')
    ax.bar(x+.18,[idx.loc[b,metric] for a,b in pairs],.36,label='Full')
    ax.set_xticks(x,['Balanced LR','RoBERTa Focal']); ax.set_ylabel(metric)
    if metric=='Macro AP': ax.set_ylim(0,1)
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGURES_DIR/'F02_scale_comparison.png',dpi=300,bbox_inches='tight'); plt.close(fig)

assert calculate_sha256(REGISTRY_PATH)==registry9_hash
assert calculate_sha256(DECISION_LOG_PATH)==decision9_hash
analysis9={'selected_run_id':selected9,'selection_partition':'validation','primary_metric':'macro_average_precision',
    'completed_runs':19,'registry_sha256':registry9_hash,'validation_dataset_sha256':EXPECTED_VALIDATION_HASH,
    'evaluation_module_sha256':evaluation_hash,'sources':sources9,'bootstrap_seed':bootstrap_seed,
    'bootstrap_replicates':B,'bootstrap_groups':len(unique_groups),'paired_with_earlier_bootstraps':False,
    'thresholds_changed':False,'statistical_significance_established':False,
    'deferred_experiments':['S03','R01','A01','A02','A03'],
    'defer_reason':'Prioritize mandatory held-out evaluation, fairness, explainability and deployment under GPU and deadline constraints.',
    'artifacts':{name:calculate_sha256(TABLES_DIR/name) for name in new_tables},
    'official_test_used':False,'final_project_model_frozen':False}
save_json_atomic(RUN_CONFIGS_DIR/'final_validation_analysis.json',analysis9)
print('\nCell 9 summary')
print('Reconstructed experiments:',len(ranking9))
print('Selected model:',selected9)
print('Validation Macro AP:',ranking9.iloc[0]['Macro AP'])
print('S02 bootstrap records:',len(boot9))
print('Original thresholds changed: No')
print('Deferred extensions: S03, R01, A01, A02, A03')
print('Model training and GPU inference: None')
print('Registry and decision log: Unchanged')
print('Official test dataset: Not loaded')
print('Model freeze: Pending Cell 10')
print('Cell 9 completed successfully.')
