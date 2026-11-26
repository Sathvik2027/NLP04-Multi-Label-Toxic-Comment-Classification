# Notebook 07, Cell 8: final evaluation verification and handoff
import gc
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from PIL import Image

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 first.'
assert len(EVALUATION_ROSTER) == len(set(EVALUATION_ROSTER)) == 19
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
verify_frozen_file(frozen_manifest['evaluation_module'])

# Verify the completed stage summaries and their dependency chain.
inventory8 = {}
def record8(path, expected_hash=None, expected_rows=None):
    path = Path(path).resolve()
    assert path.is_relative_to(PROJECT_ROOT.resolve()) and path.is_file(), path
    key = str(path.relative_to(PROJECT_ROOT.resolve()))
    digest = calculate_sha256(path) if key not in inventory8 else inventory8[key]['sha256']
    if expected_hash is not None:
        assert digest == expected_hash, f'Fingerprint mismatch: {key}'
    if expected_rows is not None:
        assert len(pd.read_csv(path)) == expected_rows, key
    inventory8[key] = {'path':key, 'sha256':digest, 'size_bytes':path.stat().st_size}
    return inventory8[key]

def summary8(path):
    s = read_json(path)
    assert s['status'] == 'Completed', path
    assert s['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH, path
    assert s['test_dataset_sha256'] == EXPECTED_TEST_HASH, path
    assert not s['thresholds_changed'], path
    assert not s.get('test_driven_selection_performed', False), path
    assert not s['model_training_performed'], path
    record8(path)
    return s

mp8 = LOGS_DIR/'test_metrics_summary.json'
fp8 = LOGS_DIR/'test_figures_summary.json'
bp8 = LOGS_DIR/'test_bootstrap_summary.json'
m8, f8, b8 = [summary8(p) for p in (mp8, fp8, bp8)]
for s in (m8, f8, b8):
    assert s['primary_selected_run_id'] == PRIMARY_RUN_ID
assert f8['metrics_summary_sha256'] == b8['metrics_summary_sha256'] == calculate_sha256(mp8)
assert b8['figures_summary_sha256'] == calculate_sha256(fp8)
assert m8['evaluated_runs'] == list(EVALUATION_ROSTER)
assert m8['test_samples_per_run'] == EXPECTED_TEST_SAMPLES == 63978
assert m8['aggregate_records'] == 38 and m8['per_label_records'] == 228
for rec in m8['artifacts'] + m8['inference_sources'] + [f8['matched_comparison_table']] + b8['artifacts']:
    record8(PROJECT_ROOT/rec['path'], rec['sha256'], rec.get('rows'))

# Verify saved input and inference records without loading trained models.
setup8 = RUN_CONFIGS_DIR/'evaluation_setup.json'
assert read_json(setup8) == evaluation_setup
record8(setup8)
record8(TABLES_DIR/'test_bootstrap_groups.csv', evaluation_setup['bootstrap_group_manifest_sha256'], EXPECTED_TEST_SAMPLES)
assert b8['group_manifest_sha256'] == evaluation_setup['bootstrap_group_manifest_sha256']
assert b8['test_groups'] == TEST_GROUP_COUNT == 63939
assert b8['test_samples'] == EXPECTED_TEST_SAMPLES
assert b8['shared_draws_across_models'] and b8['resampling_unit'] == 'normalized_text_group'
assert b8['operating_point'] == 'frozen_optimized'
tok8 = RUN_CONFIGS_DIR/'transformer_test_tokenization.json'
t8 = summary8(tok8)
assert t8['test_samples'] == EXPECTED_TEST_SAMPLES and t8['transformer_runs'] == 7
assert set(t8['run_cache_map']) == set(TRANSFORMER_RUN_IDS)
assert len(t8['caches']) == t8['tokenizer_groups']
for rec in t8['caches']:
    record8(PROJECT_ROOT/rec['cache_path'], rec['cache_sha256'])
    record8(PROJECT_ROOT/rec['cache_manifest_path'], rec['cache_manifest_sha256'])

runs8 = {}
for kind, ids in [('classical', CLASSICAL_RUN_IDS), ('transformer', TRANSFORMER_RUN_IDS)]:
    s = summary8(LOGS_DIR/f'{kind}_test_inference_summary.json')
    assert s['completed_runs'] == len(ids) and s['test_samples_per_run'] == EXPECTED_TEST_SAMPLES
    assert len(s['runs']) == len(ids) and {r['run_id'] for r in s['runs']} == set(ids)
    for rec in s['runs']:
        rid = rec['run_id']
        assert rid not in runs8 and rec['kind'] == kind
        assert rec['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
        assert rec['test_dataset_sha256'] == EXPECTED_TEST_HASH
        assert rec['test_samples'] == EXPECTED_TEST_SAMPLES
        assert not rec['model_training_performed'] and not rec['thresholds_changed']
        assert read_json(PREDICTIONS_DIR/rid/'inference_summary.json') == rec
        record8(PREDICTIONS_DIR/rid/'inference_summary.json')
        if kind == 'transformer':
            assert rec['model_state_sha256'] == FROZEN_MODELS[rid]['model_state_sha256']
            assert rec['tokenization_configuration_sha256'] == calculate_sha256(tok8)
        runs8[rid] = rec
assert set(runs8) == set(EVALUATION_ROSTER)

# Independently reconstruct AP and confusion-based metrics from the saved scores.
index8 = pd.read_csv(TABLES_DIR/'test_prediction_index.csv').set_index('Run ID')
aggregate8 = pd.read_csv(TABLES_DIR/'test_aggregate_metrics.csv').set_index(['Run ID','Operating Point'])
labels8 = pd.read_csv(TABLES_DIR/'test_per_label_metrics.csv').set_index(['Run ID','Operating Point','Label'])
assert index8.index.is_unique and set(index8.index) == set(EVALUATION_ROSTER)
assert aggregate8.index.is_unique and len(aggregate8) == 38
assert labels8.index.is_unique and len(labels8) == 228

def close8(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

verification8 = []
for rid in EVALUATION_ROSTER:
    rec, entry, ix = runs8[rid], FROZEN_MODELS[rid], index8.loc[rid]
    assert rec['predictions_path'] == ix['Prediction Path']
    assert rec['predictions_sha256'] == ix['SHA256']
    assert rec['score_type'] == ix['Score Type'] == entry['score_type']
    path = PREDICTIONS_DIR/rid/'test_predictions.parquet'
    assert path == PROJECT_ROOT/rec['predictions_path']
    record8(path, rec['predictions_sha256'])
    frame = pd.read_parquet(path)
    assert len(frame) == int(ix['Rows']) == EXPECTED_TEST_SAMPLES and frame['id'].is_unique
    assert frame['run_id'].eq(rid).all() and frame['split'].eq('test_official').all()
    np.testing.assert_array_equal(frame['id'].astype(str).to_numpy(), test_ids)
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
    scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    assert np.isfinite(scores).all()
    if entry['score_type'] != 'signed_decision_margin':
        assert ((scores >= 0) & (scores <= 1)).all()
    ap = average_precision_score(y_test, scores, average=None)
    micro_ap = average_precision_score(y_test, scores, average='micro')
    for op, thresholds in [('default',np.full(len(LABELS),entry['default_threshold'])),
                           ('optimized',np.array([entry['optimized_thresholds'][l] for l in LABELS]))]:
        pred = scores >= thresholds
        np.testing.assert_array_equal(frame[[f'pred_{op}_{l}' for l in LABELS]].to_numpy(), pred)
        tp = (pred & (y_test == 1)).sum(axis=0)
        tn = (~pred & (y_test == 0)).sum(axis=0)
        fp = (pred & (y_test == 0)).sum(axis=0)
        fn = (~pred & (y_test == 1)).sum(axis=0)
        per = labels8.loc[(rid,op),:].loc[LABELS]
        close8(per['Threshold'],thresholds)
        for name, val in [('TP',tp),('TN',tn),('FP',fp),('FN',fn)]:
            np.testing.assert_array_equal(per[name],val)
        f1 = 2*tp/(2*tp+fp+fn)
        close8(per['F1'], f1)
        close8(per['Recall'], tp/(tp+fn))
        close8(per['Average Precision'], ap)
        row = aggregate8.loc[(rid,op)]
        close8(row['Macro AP'], ap.mean())
        close8(row['Micro AP'], micro_ap)
        close8(row['Macro F1'], f1.mean())
        close8(row['Micro F1'], 2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()))
        close8(row['Hamming Loss'], np.mean(pred != y_test))
        close8(row['Exact Match Accuracy'], np.mean(np.all(pred == y_test,axis=1)))
    verification8.append({'Run ID':rid,'Test Samples':len(frame),'ID Alignment':'Passed','Frozen Decisions':'Passed','Metrics Replay':'Passed','Prediction SHA256':rec['predictions_sha256']})
    print('Final verification:',rid,flush=True)
    del frame,scores,pred
    gc.collect()

# Check the saved bootstrap replicates and percentile intervals without resampling.
reps8 = pd.read_csv(TABLES_DIR/'test_bootstrap_replicates.csv')
ci8 = pd.read_csv(TABLES_DIR/'test_bootstrap_intervals.csv')
pairs8 = pd.read_csv(TABLES_DIR/'test_paired_bootstrap_intervals.csv')
assert b8['bootstrap_replicates'] == 1000 and len(b8['run_ids']) == 4
assert len(reps8)==4000 and len(ci8)==84 and len(pairs8)==9
assert not reps8.duplicated(['Run ID','Replicate']).any()
assert set(reps8['Run ID']) == set(b8['run_ids'])
for rec in b8['prediction_sources']:
    assert rec['path'] == runs8[rec['run_id']]['predictions_path']
    assert rec['sha256'] == runs8[rec['run_id']]['predictions_sha256']

def verify_interval8(row,values):
    valid = np.asarray(values,dtype=float)
    valid = valid[np.isfinite(valid)]
    assert len(valid)==int(row['Valid Replicates'])
    assert int(row['Total Replicates'])==1000
    close8([row['Lower 2.5%'],row['Upper 97.5%']],np.percentile(valid,[2.5,97.5]))

for _, row in ci8.iterrows():
    verify_interval8(row,reps8.loc[reps8['Run ID'].eq(row['Run ID']),row['Metric']])
for _, row in pairs8.iterrows():
    a = reps8[reps8['Run ID'].eq(row['Candidate Run'])].set_index('Replicate')[row['Metric']].sort_index()
    z = reps8[reps8['Run ID'].eq(row['Reference Run'])].set_index('Replicate')[row['Metric']].sort_index()
    assert a.index.equals(z.index)
    verify_interval8(row,a-z)

# Verify the six planned publication figures, including their saved resolution.
figs8 = f8['figures']+[b8['figure']]
assert len(figs8)==6 and len({r['path'] for r in figs8})==6
for rec in figs8:
    path = PROJECT_ROOT/rec['path']
    record8(path,rec['sha256'])
    with Image.open(path) as im:
        assert im.format=='PNG' and rec['dpi']==300
        assert all(abs(v-300)<1 for v in im.info['dpi'])
        im.verify()

# Record the final handoff without altering the immutable freeze or prior summaries.
verify_path8=TABLES_DIR/'test_final_verification.csv'
save_csv_atomic(verify_path8,pd.DataFrame(verification8))
record8(verify_path8,expected_rows=19)
for name in ['official_test_label_support.csv','frozen_evaluation_roster.csv','classical_inference_summary.csv','transformer_inference_summary.csv']:
    record8(TABLES_DIR/name)
record8(FROZEN_MANIFEST_PATH,EXPECTED_FROZEN_HASH)
record8(REGISTRY_PATH,frozen_manifest['registry']['sha256'])
record8(DECISION_LOG_PATH,frozen_manifest['decision_log_sha256'])
record8(SRC_DIR/'evaluation_utils.py',frozen_manifest['evaluation_module']['sha256'])
primary8=aggregate8.loc[(PRIMARY_RUN_ID,'optimized')]
primary_ci8=ci8[(ci8['Run ID']==PRIMARY_RUN_ID)&ci8['Metric'].isin(['Macro AP','Macro F1','Micro F1'])]
for key in ['Macro AP','Macro F1','Micro F1']:
    close8(primary_ci8.loc[primary_ci8['Metric'].eq(key),'Estimate'].iloc[0],primary8[key])
assert calculate_sha256(FROZEN_MANIFEST_PATH)==EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH)==frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH)==frozen_manifest['decision_log_sha256']

completion8={'status':'Completed','notebook':'07_final_evaluation','frozen_manifest_path':str(FROZEN_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
    'frozen_manifest_sha256':EXPECTED_FROZEN_HASH,'test_dataset_path':str(TEST_PATH.relative_to(PROJECT_ROOT)),
    'test_dataset_sha256':EXPECTED_TEST_HASH,'primary_selected_run_id':PRIMARY_RUN_ID,'labels':LABELS,
    'selected_models':frozen_manifest['selected_models'],'evaluation_roster':list(EVALUATION_ROSTER),'evaluated_configurations':19,
    'test_samples_per_run':EXPECTED_TEST_SAMPLES,'test_groups':TEST_GROUP_COUNT,'aggregate_metric_records':38,'per_label_metric_records':228,
    'verified_prediction_artifacts':19,'verified_figures':6,'bootstrap_replicates':1000,'bootstrap_model_count':4,
    'bootstrap_interval_records':84,'paired_interval_records':9,'primary_test_macro_ap':float(primary8['Macro AP']),
    'primary_test_macro_roc_auc':float(primary8['Macro ROC-AUC']),'primary_test_optimized_macro_f1':float(primary8['Macro F1']),
    'primary_test_optimized_micro_f1':float(primary8['Micro F1']),
    'primary_confidence_intervals':{r['Metric']:{'lower':float(r['Lower 2.5%']),'upper':float(r['Upper 97.5%'])} for _,r in primary_ci8.iterrows()},
    'prediction_index_path':str((TABLES_DIR/'test_prediction_index.csv').relative_to(PROJECT_ROOT)),
    'prediction_index_sha256':calculate_sha256(TABLES_DIR/'test_prediction_index.csv'),
    'artifact_inventory':list(inventory8.values()),'official_test_evaluated':True,'final_project_model_frozen':True,
    'thresholds_changed':False,'test_driven_selection_performed':False,'additional_inference_performed':False,'model_training_performed':False,
    'next_notebook':'08_error_analysis'}
completion_path8=LOGS_DIR/'evaluation_completion_summary.json'
if completion_path8.exists():
    old=read_json(completion_path8)
    assert {k:v for k,v in old.items() if k!='completed_at'}==completion8,'Existing completion record differs; inspect before overwriting.'
    completion8=old
else:
    completion8['completed_at']=datetime.now(timezone.utc).isoformat()
    save_json_atomic(completion_path8,completion8)
print('\nNotebook 07 completion summary')
for k,v in completion8.items():
    if k not in ('artifact_inventory','evaluation_roster','selected_models','labels'):
        print(f'{k}: {v}')
print('Verified inventory files:',len(inventory8))
print('Completion manifest:',completion_path8.relative_to(PROJECT_ROOT))
print('Cell 8 completed successfully.')
