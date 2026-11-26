# Notebook 07, Cell 5: consolidated frozen test metrics
import gc
import numpy as np
import pandas as pd
from scipy.special import expit

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 first.'
assert len(EVALUATION_ROSTER) == 19 and len(set(EVALUATION_ROSTER)) == 19
assert len(test_df) == EXPECTED_TEST_SAMPLES == 63978
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
verify_frozen_file(frozen_manifest['evaluation_module'])
np.testing.assert_array_equal(test_df['id'].astype(str).to_numpy(), test_ids)
np.testing.assert_array_equal(test_df[LABELS].to_numpy(), y_test)

# Restore inference records rather than reloading models.
records5 = {}
source_summaries5 = []
for kind, run_ids in [('classical', CLASSICAL_RUN_IDS), ('transformer', TRANSFORMER_RUN_IDS)]:
    path = LOGS_DIR / f'{kind}_test_inference_summary.json'
    summary = read_json(path)
    assert summary['status'] == 'Completed'
    assert summary['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
    assert summary['test_dataset_sha256'] == EXPECTED_TEST_HASH
    assert summary['completed_runs'] == len(run_ids)
    assert summary['test_samples_per_run'] == EXPECTED_TEST_SAMPLES
    assert not summary['model_training_performed'] and not summary['thresholds_changed']
    assert len(summary['runs']) == len(run_ids)
    assert {rec['run_id'] for rec in summary['runs']} == set(run_ids)
    for rec in summary['runs']:
        assert rec['run_id'] not in records5
        records5[rec['run_id']] = rec
    source_summaries5.append({'path': str(path.relative_to(PROJECT_ROOT)), 'sha256': calculate_sha256(path)})
assert set(records5) == set(EVALUATION_ROSTER)
tokenization_hash5 = calculate_sha256(RUN_CONFIGS_DIR / 'transformer_test_tokenization.json')


def read_verified_test_scores5(rid):
    entry = FROZEN_MODELS[rid]
    rec = records5[rid]
    assert entry['labels'] == LABELS and entry['comparison_operator'] == '>='
    assert rec['kind'] == entry['kind'] and rec['score_type'] == entry['score_type']
    assert rec['test_samples'] == EXPECTED_TEST_SAMPLES
    assert rec['test_dataset_sha256'] == EXPECTED_TEST_HASH
    assert rec['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
    assert not rec['model_training_performed'] and not rec['thresholds_changed']
    root = PREDICTIONS_DIR / rid
    path = root / 'test_predictions.parquet'
    assert rec['predictions_path'] == str(path.relative_to(PROJECT_ROOT))
    assert read_json(root / 'inference_summary.json') == rec
    assert calculate_sha256(path) == rec['predictions_sha256']
    frame = pd.read_parquet(path)
    assert len(frame) == EXPECTED_TEST_SAMPLES and frame['id'].is_unique
    assert frame['run_id'].eq(rid).all() and frame['split'].eq('test_official').all()
    np.testing.assert_array_equal(frame['id'].astype(str).to_numpy(), test_ids)
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
    scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    assert np.isfinite(scores).all()
    if entry['score_type'] != 'signed_decision_margin':
        assert (scores >= 0).all() and (scores <= 1).all()
    if entry['kind'] == 'transformer':
        assert rec['model_state_sha256'] == entry['model_state_sha256']
        assert rec['tokenization_configuration_sha256'] == tokenization_hash5
        logits = frame[[f'logit_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
        assert np.isfinite(logits).all()
        np.testing.assert_allclose(scores, expit(logits), rtol=0, atol=2e-7)
    for name, threshold in [('default', entry['default_threshold']),
                            ('optimized', [entry['optimized_thresholds'][l] for l in LABELS])]:
        np.testing.assert_array_equal(frame[[f'pred_{name}_{l}' for l in LABELS]].to_numpy(),
                                      (scores >= np.asarray(threshold, dtype=np.float64)).astype(np.int8))
    return scores


# Compute metrics from the frozen scores at both operating points.
overall5, per_label5, cost5, index5 = [], [], [], []
for rank, rid in enumerate(EVALUATION_ROSTER, 1):
    entry = FROZEN_MODELS[rid]
    rec = records5[rid]
    scores = read_verified_test_scores5(rid)
    metadata = {'Validation Rank': rank, 'Run ID': rid, 'Model': entry['model'],
                'Kind': entry['kind'], 'Training Samples': entry['training_samples'],
                'Score Type': entry['score_type'], 'Primary Selected Model': rid == PRIMARY_RUN_ID}
    rank_metrics = None
    for name, cutoff in [('default', entry['default_threshold']),
                         ('optimized', [entry['optimized_thresholds'][l] for l in LABELS])]:
        metrics, labels = evaluate_multilabel(y_test, scores, LABELS, thresholds=cutoff)
        assert labels['Label'].tolist() == LABELS
        pred = (scores >= np.asarray(cutoff)).astype(np.int8)
        tp = ((pred == 1) & (y_test == 1)).sum(axis=0)
        tn = ((pred == 0) & (y_test == 0)).sum(axis=0)
        fp = ((pred == 1) & (y_test == 0)).sum(axis=0)
        fn = ((pred == 0) & (y_test == 1)).sum(axis=0)
        for col, expected in [('TP', tp), ('TN', tn), ('FP', fp), ('FN', fn)]:
            np.testing.assert_array_equal(labels[col].to_numpy(), expected)
        np.testing.assert_array_equal(tp + tn + fp + fn, np.full(len(LABELS), EXPECTED_TEST_SAMPLES))
        np.testing.assert_array_equal(tp + fn, y_test.sum(axis=0))
        np.testing.assert_array_equal(tn + fp, (1-y_test).sum(axis=0))
        np.testing.assert_allclose(metrics['Macro F1'], np.mean(2*tp/(2*tp+fp+fn)), rtol=0, atol=1e-12)
        np.testing.assert_allclose(metrics['Micro F1'], 2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()), rtol=0, atol=1e-12)
        np.testing.assert_allclose(metrics['Hamming Loss'], np.mean(pred != y_test), rtol=0, atol=1e-12)
        np.testing.assert_allclose(metrics['Exact Match Accuracy'], np.mean(np.all(pred == y_test, axis=1)), rtol=0, atol=1e-12)
        labels = labels.copy()
        labels['FPR'] = fp/(fp+tn)
        labels['FNR'] = fn/(fn+tp)
        labels['FPR (%)'] = labels['FPR']*100
        labels['FNR (%)'] = labels['FNR']*100
        current_ranking = np.array([metrics[k] for k in ['Macro AP','Micro AP','Macro ROC-AUC','Micro ROC-AUC']])
        if rank_metrics is None:
            rank_metrics = current_ranking
        else:
            np.testing.assert_allclose(current_ranking, rank_metrics, rtol=0, atol=1e-12)
        overall5.append({**metadata, 'Operating Point': name, 'Test Samples': EXPECTED_TEST_SAMPLES,
                         **metrics, 'Macro FPR': float(labels['FPR'].mean()), 'Macro FNR': float(labels['FNR'].mean()),
                         'Flagged Comments': int(pred.any(axis=1).sum()),
                         'Flagged Per 1000 Comments': float(pred.any(axis=1).mean()*1000)})
        per_label5.extend([{**metadata, 'Operating Point': name, **row} for row in labels.to_dict('records')])
    cost5.append({**metadata, 'Training Seconds': entry['training_time_seconds'],
                  'Inference Seconds': rec['inference_seconds'], 'Inference Device': rec['device'],
                  'Model Size MiB': rec['model_size_bytes']/1024**2,
                  'Comments Per Second (Inference Only)': EXPECTED_TEST_SAMPLES/rec['inference_seconds'],
                  'Peak GPU Allocated GiB': rec.get('peak_gpu_allocated_gib'),
                  'Peak GPU Reserved GiB': rec.get('peak_gpu_reserved_gib'),
                  'Batch Size': rec.get('batch_size', 2048),
                  'Shared Feature Transformation Seconds': rec.get('feature_transformation_seconds_shared_group'),
                  'Shared Feature Group Runs': '|'.join(rec.get('feature_group_runs', []))})
    index5.append({'Run ID': rid, 'Prediction Path': rec['predictions_path'], 'SHA256': rec['predictions_sha256'],
                   'Rows': EXPECTED_TEST_SAMPLES, 'Score Type': rec['score_type']})
    print('Verified and evaluated:', rid, flush=True)
    del scores, pred
    gc.collect()

overall5 = pd.DataFrame(overall5)
per_label5 = pd.DataFrame(per_label5)
cost5 = pd.DataFrame(cost5)
index5 = pd.DataFrame(index5)
assert len(overall5) == 38 and len(per_label5) == 228 and len(cost5) == 19
assert not overall5.duplicated(['Run ID','Operating Point']).any()
assert not per_label5.duplicated(['Run ID','Operating Point','Label']).any()

# Compare default and tuned operating points without changing either cutoff.
keys = ['Run ID','Model','Validation Rank']
metrics_compare = ['Macro F1','Micro F1','Macro Precision','Macro Recall','Hamming Loss','Exact Match Accuracy','Macro FPR','Macro FNR','Flagged Per 1000 Comments']
default5 = overall5.loc[overall5['Operating Point'].eq('default'), keys+metrics_compare]
optimized5 = overall5.loc[overall5['Operating Point'].eq('optimized'), keys+metrics_compare]
threshold_effect5 = default5.merge(optimized5, on=keys, suffixes=(' Default',' Optimized'), validate='one_to_one')
for metric in metrics_compare:
    threshold_effect5[f'{metric} Change'] = threshold_effect5[f'{metric} Optimized'] - threshold_effect5[f'{metric} Default']
label_keys = ['Run ID','Label']
label_metrics = ['Threshold','Precision','Recall','F1','FPR','FNR','TP','TN','FP','FN']
label_effect5 = per_label5.loc[per_label5['Operating Point'].eq('default'), label_keys+label_metrics].merge(
    per_label5.loc[per_label5['Operating Point'].eq('optimized'), label_keys+label_metrics],
    on=label_keys, suffixes=(' Default',' Optimized'), validate='one_to_one')
for metric in label_metrics:
    label_effect5[f'{metric} Change'] = label_effect5[f'{metric} Optimized']-label_effect5[f'{metric} Default']
confusion5 = per_label5[['Run ID','Operating Point','Label','Threshold','TP','TN','FP','FN','Positive Support','Negative Support','FPR','FNR']].copy()
assert len(threshold_effect5) == 19 and len(label_effect5) == 114

# Save comparison in the original validation rank order, not a new selection order.
comparison5 = overall5.loc[overall5['Operating Point'].eq('optimized')].copy()
comparison5 = comparison5.merge(overall5.loc[overall5['Operating Point'].eq('default'), ['Run ID','Macro F1','Micro F1']].rename(
    columns={'Macro F1':'Default Macro F1','Micro F1':'Default Micro F1'}), on='Run ID', validate='one_to_one')
comparison5 = comparison5.rename(columns={'Macro F1':'Optimized Macro F1','Micro F1':'Optimized Micro F1'})
comparison5['Validation Macro AP'] = comparison5['Run ID'].map(lambda rid: FROZEN_MODELS[rid]['validation_macro_ap'])
comparison5['Validation Optimized Macro F1'] = comparison5['Run ID'].map(lambda rid: FROZEN_MODELS[rid]['validation_optimized_macro_f1'])
comparison5 = comparison5.sort_values('Validation Rank').reset_index(drop=True)
assert comparison5.iloc[0]['Run ID'] == PRIMARY_RUN_ID

outputs5 = {'test_aggregate_metrics.csv':overall5,'test_per_label_metrics.csv':per_label5,
            'test_confusion_counts.csv':confusion5,'test_threshold_effect.csv':threshold_effect5,
            'test_per_label_threshold_effect.csv':label_effect5,'test_model_comparison.csv':comparison5,
            'test_runtime_model_size.csv':cost5,'test_prediction_index.csv':index5}
for name, frame in outputs5.items():
    path = TABLES_DIR / name
    save_csv_atomic(path, frame)
    assert len(pd.read_csv(path)) == len(frame)
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
primary5 = comparison5.loc[comparison5['Run ID'].eq(PRIMARY_RUN_ID)].iloc[0]
summary5 = {'status':'Completed','frozen_manifest_sha256':EXPECTED_FROZEN_HASH,
            'test_dataset_sha256':EXPECTED_TEST_HASH,'primary_selected_run_id':PRIMARY_RUN_ID,
            'evaluated_runs':list(EVALUATION_ROSTER),'test_samples_per_run':EXPECTED_TEST_SAMPLES,
            'aggregate_records':len(overall5),'per_label_records':len(per_label5),
            'primary_test_macro_ap':float(primary5['Macro AP']),
            'primary_test_optimized_macro_f1':float(primary5['Optimized Macro F1']),
            'primary_test_optimized_micro_f1':float(primary5['Optimized Micro F1']),
            'inference_sources':source_summaries5,
            'artifacts':[{'path':str((TABLES_DIR/name).relative_to(PROJECT_ROOT)),
                          'sha256':calculate_sha256(TABLES_DIR/name),'rows':len(frame)} for name,frame in outputs5.items()],
            'test_driven_selection_performed':False,'thresholds_changed':False,
            'model_training_performed':False,'additional_inference_performed':False,
            'bootstrap_intervals_completed':False}
save_json_atomic(LOGS_DIR/'test_metrics_summary.json', summary5)
print('\nOfficial test comparison (ordered by frozen validation rank)')
display(comparison5[['Validation Rank','Run ID','Training Samples','Validation Macro AP','Macro AP','Macro ROC-AUC',
                     'Default Macro F1','Optimized Macro F1','Optimized Micro F1','Hamming Loss','Exact Match Accuracy']].round(6))
print('\nFrozen primary model: per-label official test metrics')
display(per_label5.loc[(per_label5['Run ID'].eq(PRIMARY_RUN_ID)) & (per_label5['Operating Point'].eq('optimized')),
                       ['Label','Threshold','Precision','Recall','F1','Average Precision','ROC-AUC','Positive Support','TP','TN','FP','FN','FPR (%)','FNR (%)']].round(6))
print('\nPrimary model: default versus validation-tuned thresholds')
display(threshold_effect5.loc[threshold_effect5['Run ID'].eq(PRIMARY_RUN_ID),
                             ['Run ID','Macro F1 Default','Macro F1 Optimized','Macro F1 Change','Micro F1 Default','Micro F1 Optimized','Micro F1 Change']].round(6))
print('\nEvaluated configurations:', len(EVALUATION_ROSTER))
print('Aggregate metric records:',len(overall5))
print('Per-label metric records:',len(per_label5))
print('Prediction artifacts verified:',len(index5))
print('Primary selected model:',PRIMARY_RUN_ID)
print('Additional inference or model training: None')
print('Thresholds and frozen manifest: Unchanged')
print('Cell 5 completed successfully.')
