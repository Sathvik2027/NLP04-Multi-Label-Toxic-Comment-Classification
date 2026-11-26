# Notebook 07, Cell 7: paired group bootstrap from saved test scores
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 first.'
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
metrics_path7 = LOGS_DIR / 'test_metrics_summary.json'
figures_path7 = LOGS_DIR / 'test_figures_summary.json'
metrics_hash7 = calculate_sha256(metrics_path7)
figures_hash7 = calculate_sha256(figures_path7)
metrics7 = read_json(metrics_path7)
figures7 = read_json(figures_path7)
for summary in (metrics7, figures7):
    assert summary['status'] == 'Completed'
    assert summary['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
    assert summary['test_dataset_sha256'] == EXPECTED_TEST_HASH
    assert summary['primary_selected_run_id'] == PRIMARY_RUN_ID
    assert not summary['thresholds_changed']
    assert not summary['test_driven_selection_performed']
assert figures7['metrics_summary_sha256'] == metrics_hash7
for rec in metrics7['artifacts'] + figures7['figures'] + [figures7['matched_comparison_table']]:
    assert calculate_sha256(PROJECT_ROOT / rec['path']) == rec['sha256'], rec['path']

# Use the saved normalized-text grouping and shared bootstrap draws.
group_path7 = TABLES_DIR / 'test_bootstrap_groups.csv'
group_hash7 = calculate_sha256(group_path7)
assert group_hash7 == evaluation_setup['bootstrap_group_manifest_sha256']
group_frame7 = pd.read_csv(group_path7, dtype={'id':str})
np.testing.assert_array_equal(group_frame7['id'].to_numpy(), test_ids)
np.testing.assert_array_equal(group_frame7['row_index'].to_numpy(), np.arange(EXPECTED_TEST_SAMPLES))
groups7 = group_frame7['bootstrap_group_id'].to_numpy(dtype=np.int64)
np.testing.assert_array_equal(groups7, test_group_ids)
G7 = len(np.unique(groups7))
assert G7 == TEST_GROUP_COUNT == 63939
np.testing.assert_array_equal(np.unique(groups7), np.arange(G7))
assert len(y_test) == EXPECTED_TEST_SAMPLES == 63978

RUNS7 = [PRIMARY_RUN_ID, 'T06_FOCAL_S42', 'T03_FOCAL_S42', 'S01_LR_COMBINED_BALANCED_FULL_S42']
assert len(set(RUNS7)) == 4 and set(RUNS7).issubset(EVALUATION_ROSTER)
B7 = 1000
BOOTSTRAP_SEED7 = 42
index7 = pd.read_csv(TABLES_DIR / 'test_prediction_index.csv').set_index('Run ID')
aggregate7 = pd.read_csv(TABLES_DIR / 'test_aggregate_metrics.csv')
aggregate7 = aggregate7[aggregate7['Operating Point'].eq('optimized')].set_index('Run ID')
labels7 = pd.read_csv(TABLES_DIR / 'test_per_label_metrics.csv')
labels7 = labels7[labels7['Operating Point'].eq('optimized')]

# Pre-sort once. Equal scores are treated as one threshold, including ties.
def prepare_ap7(scores, truth, group_ids):
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(order)-1]
    return group_ids[order], truth[order].astype(np.int64), ends


def weighted_ap7(prepared, group_weights):
    group_order, truth, ends = prepared
    w = group_weights[group_order]
    positive = np.cumsum(w * truth, dtype=np.int64)[ends]
    total = np.cumsum(w, dtype=np.int64)[ends]
    if positive[-1] == 0 or total[-1] == positive[-1]:
        return np.nan
    precision = np.divide(positive, total, out=np.zeros(len(total), dtype=float), where=total > 0)
    return float(np.dot(np.diff(positive, prepend=0).astype(float), precision) / positive[-1])

# Group-level counts preserve all six labels jointly.
def group_counts7(mask, group_ids, n_groups):
    return np.column_stack([np.bincount(group_ids, weights=mask[:,j], minlength=n_groups) for j in range(mask.shape[1])]).astype(np.int64)


def score_bootstrap7(prepared, counts, weights, positives):
    tp = weights @ counts[0]
    fp = weights @ counts[1]
    neg = int(weights @ counts[2]) - positives
    valid = (positives > 0) & (neg > 0)
    f1 = np.divide(2*tp, positives+tp+fp, out=np.full(len(LABELS), np.nan), where=valid)
    recall = np.divide(tp, positives, out=np.full(len(LABELS), np.nan), where=valid)
    ap = np.array([weighted_ap7(p, weights) for p in prepared])
    macro_ap = float(ap.mean()) if np.isfinite(ap).all() else np.nan
    macro_f1 = float(f1.mean()) if np.isfinite(f1).all() else np.nan
    micro_f1 = float(2*tp.sum()/(positives.sum()+tp.sum()+fp.sum())) if valid.all() else np.nan
    return np.r_[macro_ap, macro_f1, micro_f1, ap, f1, recall]

metric_names7 = ['Macro AP','Macro F1','Micro F1'] + [f'AP: {l}' for l in LABELS] + [f'F1: {l}' for l in LABELS] + [f'Recall: {l}' for l in LABELS]
positive_counts7 = group_counts7(y_test, groups7, G7)
group_sizes7 = np.bincount(groups7, minlength=G7).astype(np.int64)
prepared7, counts7, point7, sources7, scores_for_checks7 = {}, {}, {}, [], {}
for rid in RUNS7:
    row = index7.loc[rid]
    path = PROJECT_ROOT / row['Prediction Path']
    assert calculate_sha256(path) == row['SHA256']
    frame = pd.read_parquet(path)
    assert len(frame) == EXPECTED_TEST_SAMPLES and frame['id'].is_unique
    assert frame['run_id'].eq(rid).all() and frame['split'].eq('test_official').all()
    np.testing.assert_array_equal(frame['id'].astype(str).to_numpy(), test_ids)
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
    scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    assert np.isfinite(scores).all()
    cutoffs = np.array([FROZEN_MODELS[rid]['optimized_thresholds'][l] for l in LABELS])
    pred = scores >= cutoffs
    np.testing.assert_array_equal(pred, frame[[f'pred_optimized_{l}' for l in LABELS]].to_numpy())
    prepared7[rid] = [prepare_ap7(scores[:,j], y_test[:,j], groups7) for j in range(len(LABELS))]
    counts7[rid] = (group_counts7(pred & (y_test==1), groups7,G7), group_counts7(pred & (y_test==0),groups7,G7), group_sizes7)
    p = labels7[labels7['Run ID'].eq(rid)].set_index('Label').loc[LABELS]
    point7[rid] = np.r_[aggregate7.loc[rid,['Macro AP','Macro F1','Micro F1']].to_numpy(dtype=float), p['Average Precision'].to_numpy(), p['F1'].to_numpy(), p['Recall'].to_numpy()]
    ones = np.ones(G7,dtype=np.int64)
    reconstructed = score_bootstrap7(prepared7[rid],counts7[rid],ones,ones @ positive_counts7)
    np.testing.assert_allclose(reconstructed,point7[rid],rtol=0,atol=1e-12)
    scores_for_checks7[rid] = scores
    sources7.append({'run_id':rid,'path':row['Prediction Path'],'sha256':str(row['SHA256'])})
    print('Verified bootstrap input:',rid,flush=True)
    del frame,pred,scores

# Check exact weighted AP against sklearn on two independent group draws.
check_rng7 = np.random.default_rng(2026)
check_count7 = 0
for _ in range(2):
    w = np.bincount(check_rng7.integers(0,G7,size=G7),minlength=G7)
    for rid in RUNS7:
        for j in range(len(LABELS)):
            expected = average_precision_score(y_test[:,j],scores_for_checks7[rid][:,j],sample_weight=w[groups7])
            actual = weighted_ap7(prepared7[rid][j],w)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            check_count7 += 1
print('Weighted AP checks passed:',check_count7)
del scores_for_checks7

# One shared draw of groups per replicate for every model.
# Limit BLAS threads to avoid overhead on short dot products.
from threadpoolctl import threadpool_limits
rng7 = np.random.default_rng(BOOTSTRAP_SEED7)
boot7 = np.full((B7,len(RUNS7),len(metric_names7)),np.nan,dtype=np.float64)
start7 = time.perf_counter()
with threadpool_limits(limits=1):
    for b in range(B7):
        weights = np.bincount(rng7.integers(0,G7,size=G7),minlength=G7)
        positives = weights @ positive_counts7
        for i,rid in enumerate(RUNS7):
            boot7[b,i] = score_bootstrap7(prepared7[rid],counts7[rid],weights,positives)
        if (b+1)%100 == 0:
            print(f'Bootstrap {b+1}/{B7} | elapsed {time.perf_counter()-start7:.1f}s',flush=True)
elapsed7 = time.perf_counter()-start7

# Keep undefined replicates as missing values, never zero-fill.
def interval7(values, estimate):
    values = np.asarray(values,dtype=float)
    valid = values[np.isfinite(values)]
    low,high = np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan)
    return {'Estimate':float(estimate),'Lower 2.5%':float(low),'Upper 97.5%':float(high),'Valid Replicates':len(valid),'Total Replicates':B7}

interval_rows7,replicate_rows7 = [],[]
for i,rid in enumerate(RUNS7):
    for j,metric in enumerate(metric_names7):
        interval_rows7.append({'Run ID':rid,'Metric':metric,**interval7(boot7[:,i,j],point7[rid][j])})
    for b in range(B7):
        replicate_rows7.append({'Replicate':b+1,'Run ID':rid,**dict(zip(metric_names7,boot7[b,i]))})
intervals7 = pd.DataFrame(interval_rows7)
replicates7 = pd.DataFrame(replicate_rows7)
assert len(intervals7)==4*21 and len(replicates7)==B7*4

# Primary model minus each validation-selected comparator on identical draws.
pair_rows7 = []
for i,rid in enumerate(RUNS7[1:],start=1):
    for j,metric in enumerate(metric_names7[:3]):
        pair_rows7.append({'Candidate Run':PRIMARY_RUN_ID,'Reference Run':rid,'Metric':metric,
                          **interval7(boot7[:,0,j]-boot7[:,i,j],point7[PRIMARY_RUN_ID][j]-point7[rid][j])})
pairs7 = pd.DataFrame(pair_rows7)
assert len(pairs7)==9

outputs7 = {'test_bootstrap_intervals.csv':intervals7,'test_bootstrap_replicates.csv':replicates7,'test_paired_bootstrap_intervals.csv':pairs7}
for name, frame in outputs7.items():
    save_csv_atomic(TABLES_DIR/name,frame)
    assert len(pd.read_csv(TABLES_DIR/name))==len(frame)

# Plot paired Macro AP and F1 differences; no new model selection.
fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
selected_pairs=pairs7[pairs7['Metric'].isin(['Macro AP','Macro F1'])].reset_index(drop=True)
short={'T06_FOCAL_S42':'Representative RoBERTa','T03_FOCAL_S42':'Representative DistilBERT','S01_LR_COMBINED_BALANCED_FULL_S42':'Full-data LR'}
for i,row in selected_pairs.iterrows():
    color='#34699a' if row['Metric']=='Macro AP' else '#31846b'
    ax.plot([row['Lower 2.5%'],row['Upper 97.5%']],[i,i],color=color,linewidth=2)
    ax.scatter(row['Estimate'],i,color=color,s=32,zorder=3)
ax.axvline(0,color='#666666',linestyle='--',linewidth=1)
ax.set_yticks(np.arange(len(selected_pairs)),[f"{short[r['Reference Run']]} — {r['Metric']}" for _,r in selected_pairs.iterrows()])
ax.invert_yaxis(); ax.set_xlabel('Frozen primary model minus reference (score difference)')
ax.set_title('Paired group-bootstrap 95% percentile intervals'); ax.grid(axis='x',alpha=.2)
figure_path7=FIGURES_DIR/'F06_paired_CI.png'
tmp7=figure_path7.with_name(figure_path7.stem+'.tmp.png')
fig.savefig(tmp7,dpi=300,bbox_inches='tight',facecolor='white'); tmp7.replace(figure_path7)
plt.show(); plt.close(fig)

assert calculate_sha256(FROZEN_MANIFEST_PATH)==EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH)==frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH)==frozen_manifest['decision_log_sha256']
assert calculate_sha256(metrics_path7)==metrics_hash7
assert calculate_sha256(figures_path7)==figures_hash7
summary7={'status':'Completed','frozen_manifest_sha256':EXPECTED_FROZEN_HASH,'test_dataset_sha256':EXPECTED_TEST_HASH,
          'primary_selected_run_id':PRIMARY_RUN_ID,'run_ids':RUNS7,'bootstrap_replicates':B7,'bootstrap_seed':BOOTSTRAP_SEED7,
          'test_samples':EXPECTED_TEST_SAMPLES,'test_groups':G7,'group_manifest_sha256':group_hash7,
          'resampling_unit':'normalized_text_group','shared_draws_across_models':True,'operating_point':'frozen_optimized',
          'interval_method':'percentile_95','weighted_ap_verification_checks':check_count7,
          'metrics_summary_sha256':metrics_hash7,'figures_summary_sha256':figures_hash7,
          'prediction_sources':sources7,'scope':'Primary and three validation-selected comparators; conditional on fixed trained models; not seed variation or multiplicity-adjusted inference.',
          'elapsed_seconds':elapsed7,'artifacts':[{'path':str((TABLES_DIR/name).relative_to(PROJECT_ROOT)),'sha256':calculate_sha256(TABLES_DIR/name),'rows':len(frame)} for name,frame in outputs7.items()],
          'figure':{'path':str(figure_path7.relative_to(PROJECT_ROOT)),'sha256':calculate_sha256(figure_path7),'dpi':300},
          'thresholds_changed':False,'test_driven_selection_performed':False,'additional_inference_performed':False,'model_training_performed':False}
save_json_atomic(LOGS_DIR/'test_bootstrap_summary.json',summary7)
print('\nPrimary model bootstrap intervals')
display(intervals7[intervals7['Run ID'].eq(PRIMARY_RUN_ID)].round(6))
print('\nPaired comparison intervals')
display(pairs7.round(6))
print('Completed bootstrap replicates:',B7)
print('Models:',len(RUNS7),'Replicate records:',len(replicates7),'Interval records:',len(intervals7),'Paired records:',len(pairs7))
print('Elapsed computation seconds:',round(elapsed7,2))
print('Frozen thresholds and manifest: Unchanged')
print('Additional inference and training: None')
print('Cell 7 completed successfully.')
gc.collect()
