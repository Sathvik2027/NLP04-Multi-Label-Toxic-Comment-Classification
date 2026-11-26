# Notebook 06, Cell 8: finalize S02 without further training
from pathlib import Path
from datetime import datetime, timezone
import gc, json, time, shutil
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

run_id = S02_RUN_ID
assert run_id == 'S02_ROBERTA_FOCAL_FULL_S42'
R = RESULTS_DIR / run_id
M = MODELS_DIR / run_id
T, L, P = R/'tables', R/'logs', R/'predictions'
config_path = R/'config.json'
completion_path = L/'training_completion_summary.json'
completion = read_json(completion_path)
configuration = read_json(config_path)
assert completion['training_completed'] and not completion['official_test_used']
assert completion['run_id'] == run_id == configuration['run_id']
assert completion['training_samples'] == 127628 and completion['validation_samples'] == 31911
assert completion['epochs_completed'] == 3 and completion['attempted_updates'] == 11967
assert completion['successful_updates'] + completion['skipped_updates'] == 11967
assert configuration['run_signature'] == get_run_signature(run_id)
assert calculate_sha256(VALIDATION_PATH) == EXPECTED_VALIDATION_HASH
assert calculate_sha256(SRC_DIR/'evaluation_utils.py') == evaluation_hash
registry_before = pd.read_csv(REGISTRY_PATH, dtype=str, keep_default_na=False)
assert registry_before['run_id'].is_unique
assert len(registry_before) in (18,19)
registered = run_id in registry_before['run_id'].values
assert registered == (len(registry_before)==19)
decision_hash_before = calculate_sha256(DECISION_LOG_PATH)
registry_hash_before = calculate_sha256(REGISTRY_PATH)
model_path = PROJECT_ROOT / completion['best_model_path']
raw_path = PROJECT_ROOT / completion['best_validation_path']
checkpoint_manifest_path = model_path/'checkpoint_manifest.json'
manifest = read_json(checkpoint_manifest_path)
assert manifest['epoch'] == completion['best_epoch'] == 3
assert manifest['labels'] == LABELS
assert manifest['model_state_sha256'] == completion['best_model_state_sha256']
assert manifest['initial_model_sha256'] == completion['initial_model_state_sha256'] == INITIAL_STATE_HASH
verify_model_files(model_path, manifest)
assert calculate_sha256(raw_path) == completion['best_validation_sha256']
rel = lambda path: str(Path(path).relative_to(PROJECT_ROOT))

# Restore selected scores and check every validation row.
with np.load(raw_path, allow_pickle=False) as data:
    ids = data['ids'].copy()
    targets = data['targets'].copy()
    logits = data['logits'].copy()
    scores = data['scores'].copy()
validation = pd.read_parquet(VALIDATION_PATH, columns=['id', *LABELS])
np.testing.assert_array_equal(ids, validation['id'].astype(str).to_numpy())
np.testing.assert_array_equal(ids, val_comment_ids)
np.testing.assert_array_equal(targets, validation[LABELS].to_numpy())
np.testing.assert_array_equal(targets, y_val)
assert scores.shape == logits.shape == targets.shape == (31911,6)
assert np.isfinite(scores).all() and np.isfinite(logits).all()
assert ((scores >= 0) & (scores <= 1)).all()
np.testing.assert_allclose(scores, torch.sigmoid(torch.from_numpy(logits)).numpy(), atol=2e-7, rtol=0)
default_metrics, default_per_label = evaluate_multilabel(targets,scores,LABELS,thresholds=0.5)
assert abs(default_metrics['Macro AP'] - completion['best_validation_macro_ap']) <= 1e-12
history = pd.read_csv(T/'epoch_history.csv')
assert history['Epoch'].tolist() == [1,2,3]
assert int(history.loc[history['Validation Macro AP'].idxmax(),'Epoch']) == completion['best_epoch']
assert abs(history['Validation Macro AP'].max()-default_metrics['Macro AP']) <= 1e-12
print('Saved model files, selected scores, IDs and epoch selection verified.')

# Reproduce the original inference once, and cache verified reload evidence.
reload_path = L/'model_reload_verification.json'
reload_binding = {'model_manifest_sha256':calculate_sha256(checkpoint_manifest_path),
                  'validation_scores_sha256':calculate_sha256(raw_path),
                  'run_config_sha256':calculate_sha256(config_path),
                  'evaluation_module_sha256':evaluation_hash}
if reload_path.is_file():
    reload_info = read_json(reload_path)
    assert reload_info['binding'] == reload_binding and reload_info['verification']=='Passed'
    assert reload_info['maximum_logit_reload_error'] <= 2e-3
    assert reload_info['maximum_probability_reload_error'] <= 5e-4
    print('Previously verified model reload evidence reused.')
else:
    assert torch.cuda.is_available(), 'GPU required for one validation reload pass; no training needed.'
    gc.collect(); torch.cuda.empty_cache()
    reload_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    assert reload_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert reload_tokenizer.pad_token_id == tokenizer.pad_token_id == 1
    assert reload_tokenizer.padding_side == reload_tokenizer.truncation_side == 'right'
    test_texts = ['Tokenizer consistency check.', 'A second example.']
    assert reload_tokenizer(test_texts)['input_ids'] == tokenizer(test_texts)['input_ids']
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True)
    assert model_state_sha256(reloaded_model) == manifest['model_state_sha256']
    assert reloaded_model.config.id2label == {i: label for i,label in enumerate(LABELS)}
    reloaded_model = reloaded_model.to(DEVICE)
    print('Running one validation reload pass...',flush=True)
    result = evaluate_validation_model(reloaded_model,'FOCAL')
    logit_error = float(np.max(np.abs(logits-result['logits'])))
    score_error = float(np.max(np.abs(scores-result['scores'])))
    ap_error = abs(float(default_metrics['Macro AP'])-float(result['metrics']['Macro AP']))
    assert logit_error <= 2e-3, f'Logit reload error: {logit_error}'
    assert score_error <= 5e-4, f'Score reload error: {score_error}'
    reload_info = {'binding':reload_binding,'verification':'Passed',
                   'model_state_exact_match':True,'tokenizer_verified':True,
                   'validation_rows':len(ids),'maximum_logit_reload_error':logit_error,
                   'maximum_probability_reload_error':score_error,'reload_macro_ap_difference':ap_error,
                   'validation_prediction_time_seconds':float(result['inference_time_seconds']),
                   'official_test_used':False}
    save_json_atomic(reload_path,reload_info)
    del reloaded_model, result, reload_tokenizer
    gc.collect(); torch.cuda.empty_cache()
print('Reload verification:',reload_info['maximum_probability_reload_error'])

# Optimize thresholds only on the original selected validation scores.
threshold_df = optimize_thresholds(targets,scores,LABELS).set_index('Label').loc[LABELS].reset_index()
thresholds = threshold_df['Optimal Threshold'].to_numpy(dtype=np.float64)
assert thresholds.shape==(6,) and np.isfinite(thresholds).all()
assert ((thresholds>=0)&(thresholds<=1)).all()
optimized_metrics, optimized_per_label = evaluate_multilabel(targets,scores,LABELS,thresholds=thresholds)
np.testing.assert_allclose(optimized_per_label['F1'],threshold_df['Validation F1'],rtol=0,atol=1e-12)
assert abs(optimized_metrics['Macro AP']-default_metrics['Macro AP']) <= 1e-12
assert optimized_metrics['Macro F1'] >= default_metrics['Macro F1']-1e-12
threshold_path = R/'thresholds.json'
threshold_config = {'run_id':run_id,'labels':LABELS,'score_type':'sigmoid_probability',
                    'default_threshold':0.5,'optimized_thresholds':dict(zip(LABELS,map(float,thresholds))),
                    'selection_partition':'validation','optimization_metric':'per_label_f1',
                    'comparison_operator':'>=','tie_breaking':'highest_threshold',
                    'probability_calibration_established':False,'official_test_used':False}
if threshold_path.exists():
    assert read_json(threshold_path)==threshold_config
else:
    save_json_atomic(threshold_path,threshold_config)

# Save ID-aligned predictions in the established format.
predictions = pd.DataFrame({'id':ids,'split':'validation','run_id':run_id})
default_pred = apply_thresholds(scores,0.5)
optimized_pred = apply_thresholds(scores,thresholds)
for i,label in enumerate(LABELS):
    predictions[f'true_{label}']=targets[:,i]
    predictions[f'logit_{label}']=logits[:,i]
    predictions[f'score_{label}']=scores[:,i]
    predictions[f'pred_default_{label}']=default_pred[:,i]
    predictions[f'pred_optimized_{label}']=optimized_pred[:,i]
prediction_path = P/'validation_predictions.parquet'
if prediction_path.exists():
    pd.testing.assert_frame_equal(pd.read_parquet(prediction_path),predictions,check_exact=True)
else:
    temp = prediction_path.with_suffix('.parquet.tmp')
    predictions.to_parquet(temp,index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(temp),predictions,check_exact=True)
    temp.replace(prediction_path)

# Reconstruct final metrics from persisted predictions.
saved = pd.read_parquet(prediction_path)
recomputed, recomputed_labels = evaluate_multilabel(saved[[f'true_{x}' for x in LABELS]].to_numpy(),
    saved[[f'score_{x}' for x in LABELS]].to_numpy(),LABELS,thresholds=thresholds)
for key in optimized_metrics:
    assert abs(float(recomputed[key])-float(optimized_metrics[key]))<=1e-12
pd.testing.assert_frame_equal(recomputed_labels,optimized_per_label,check_exact=True)
np.testing.assert_array_equal(saved[[f'pred_optimized_{x}' for x in LABELS]].to_numpy(),optimized_pred)

# Compare scaling with the original T06 using its saved validation artifacts.
ref = registry_before.set_index('run_id').loc['T06_FOCAL_S42']
ref_root = PROJECT_ROOT/ref['results_path']
ref_summary = read_json(ref_root/'logs/run_summary.json')
ref_labels = pd.read_csv(ref_root/'tables/optimized_per_label_metrics.csv').set_index('Label').loc[LABELS].reset_index()
per_label_scaling = ref_labels[['Label','Average Precision','F1']].merge(
    optimized_per_label[['Label','Average Precision','F1']],on='Label',suffixes=('_Representative','_Full'),validate='one_to_one')
for metric in ['Average Precision','F1']:
    per_label_scaling[f'{metric} Change']=per_label_scaling[f'{metric}_Full']-per_label_scaling[f'{metric}_Representative']
model_size = sum(p.stat().st_size for p in model_path.glob('*.safetensors'))/1024**2
summary = {'run_id':run_id,'reference_run_id':'T06_FOCAL_S42','model':'RoBERTa','loss':'FOCAL',
           'training_samples':127628,'validation_samples':31911,'seed':SEED,'maximum_length':MAX_LENGTH,
           'epochs_completed':completion['epochs_completed'],'best_epoch':completion['best_epoch'],
           'stopping_reason':completion['stopping_reason'],'attempted_updates':completion['attempted_updates'],
           'successful_updates':completion['successful_updates'],'skipped_updates':completion['skipped_updates'],
           'validation_macro_ap':float(default_metrics['Macro AP']),
           'validation_macro_roc_auc':float(default_metrics['Macro ROC-AUC']),
           'validation_default_macro_f1':float(default_metrics['Macro F1']),
           'validation_optimized_macro_f1':float(optimized_metrics['Macro F1']),
           'validation_default_micro_f1':float(default_metrics['Micro F1']),
           'validation_optimized_micro_f1':float(optimized_metrics['Micro F1']),
           'training_time_seconds':completion['training_time_seconds'],
           'training_time_excludes_validation':True,
           'session_wall_time_seconds':completion['current_session_wall_time_seconds'],
           'validation_prediction_time_seconds':reload_info['validation_prediction_time_seconds'],
           'model_weights_size_mib':model_size,
           'maximum_logit_reload_error':reload_info['maximum_logit_reload_error'],
           'maximum_probability_reload_error':reload_info['maximum_probability_reload_error'],
           'reload_macro_ap_difference':reload_info['reload_macro_ap_difference'],
           'best_model_state_sha256':manifest['model_state_sha256'],
           'best_model_path':rel(model_path),'threshold_path':rel(threshold_path),
           'predictions_path':rel(prediction_path),'predictions_sha256':calculate_sha256(prediction_path),
           'initial_model_state_sha256':INITIAL_STATE_HASH,
           'determinism_policy_source':configuration['determinism_policy_source'],
           'historical_determinism_policy_confirmed':configuration['historical_determinism_policy_confirmed'],
           'macro_ap_change_from_reference':float(default_metrics['Macro AP'])-float(ref_summary['validation_macro_ap']),
           'macro_f1_change_from_reference':float(optimized_metrics['Macro F1'])-float(ref_summary['validation_optimized_macro_f1']),
           'statistical_significance_established':False,'final_metrics_verified':True,
           'official_test_used':False,'final_project_model_frozen':False}
assert model_size>0
scaling_df = pd.DataFrame([{'Metric':key,'Representative':ref_summary[key],'Full':summary[key],
                            'Change':summary[key]-ref_summary[key]} for key in
    ['training_samples','validation_macro_ap','validation_macro_roc_auc','validation_default_macro_f1',
     'validation_optimized_macro_f1','validation_default_micro_f1','validation_optimized_micro_f1','training_time_seconds']])
comparison_df = pd.DataFrame([{'Metric':key,'Default':default_metrics[key],'Optimized':optimized_metrics[key],
                              'Change':optimized_metrics[key]-default_metrics[key]} for key in default_metrics])
per_label_comparison = default_per_label[['Label','Threshold','Precision','Recall','F1']].merge(
    optimized_per_label[['Label','Threshold','Precision','Recall','F1','Average Precision','ROC-AUC','Positive Support']],
    on='Label',suffixes=('_Default','_Optimized'),validate='one_to_one')
per_label_comparison['F1 Improvement']=per_label_comparison['F1_Optimized']-per_label_comparison['F1_Default']
tables = {'default_overall_metrics.csv':pd.DataFrame([default_metrics]),
          'optimized_overall_metrics.csv':pd.DataFrame([optimized_metrics]),
          'default_per_label_metrics.csv':default_per_label,'optimized_per_label_metrics.csv':optimized_per_label,
          'optimized_thresholds.csv':threshold_df,'overall_threshold_comparison.csv':comparison_df,
          'per_label_threshold_comparison.csv':per_label_comparison,'scaling_comparison.csv':scaling_df,
          'scaling_per_label_comparison.csv':per_label_scaling,'run_summary.csv':pd.DataFrame([summary])}
summary_path = L/'run_summary.json'
manifest_path = T/'artifact_manifest.csv'

# Register only after all retained artifacts have been verified.
if not registered:
    for name,frame in tables.items(): save_csv_atomic(T/name,frame)
    save_json_atomic(summary_path,summary)
    paths = [config_path,completion_path,raw_path,threshold_path,prediction_path,reload_path,summary_path,
             T/'epoch_history.csv',checkpoint_manifest_path,*[T/name for name in tables],
             *[model_path/name for name in manifest['files']]]
    assert len(set(paths))==len(paths)
    records = [{'Run ID':run_id,'Path':rel(p),'Size Bytes':p.stat().st_size,'SHA256':calculate_sha256(p),
                'Verification':'Passed'} for p in paths]
    save_csv_atomic(manifest_path,pd.DataFrame(records))
else:
    assert read_json(summary_path)==summary
records_df = pd.read_csv(manifest_path)
assert records_df['Path'].is_unique
for row in records_df.to_dict('records'):
    p=PROJECT_ROOT/row['Path']
    assert p.is_file() and p.stat().st_size==row['Size Bytes'] and calculate_sha256(p)==row['SHA256']
assert calculate_sha256(DECISION_LOG_PATH)==decision_hash_before
assert calculate_sha256(REGISTRY_PATH)==registry_hash_before
assert calculate_sha256(FULL_TRAIN_PATH)==EXPECTED_FULL_TRAINING_HASH
assert calculate_sha256(VALIDATION_PATH)==EXPECTED_VALIDATION_HASH
if not registered:
    # Cell 7 did not persist a start timestamp; do not invent one.
    record={'run_id':run_id,'notebook':NOTEBOOK_DIR.name,'model':'RoBERTa',
            'feature_representation':'contextual_embeddings','loss_function':'FOCAL',
            'sampling_strategy':'full_training_pool','training_samples':127628,'seed':SEED,'max_length':MAX_LENGTH,
            'config_path':rel(config_path),'model_path':rel(model_path),'results_path':rel(R),
            'validation_macro_ap':summary['validation_macro_ap'],'validation_macro_f1':summary['validation_optimized_macro_f1'],
            'training_time_seconds':summary['training_time_seconds'],'status':'Completed','start_time':'',
            'end_time':datetime.now(timezone.utc).isoformat(),
            'notes':'Full-data scaling from shared initialization; validation-selected thresholds; end_time is registration time; start timestamp not recorded; historical determinism policy unavailable; no official test.'}
    assert set(record)==set(registry_before.columns)
    new_registry=pd.concat([registry_before,pd.DataFrame([{c:str(record[c]) for c in registry_before.columns}])],ignore_index=True)
    save_csv_atomic(REGISTRY_PATH,new_registry)
    pd.testing.assert_frame_equal(pd.read_csv(REGISTRY_PATH,dtype=str,keep_default_na=False),new_registry,check_exact=True)
registry_after=pd.read_csv(REGISTRY_PATH,dtype=str,keep_default_na=False)
assert len(registry_after)==19 and registry_after['run_id'].is_unique
assert registry_after.loc[registry_after['run_id'].eq(run_id),'status'].tolist()==['Completed']
assert abs(float(registry_after.loc[registry_after['run_id'].eq(run_id),'validation_macro_ap'].iloc[0])-summary['validation_macro_ap'])<=1e-12

# Remove only the exact recovery files after successful registration.
cleanup_path=L/'checkpoint_retention.json'
if not cleanup_path.exists():
    removed=[]
    for p in [M/'latest_checkpoint.pt',Path('/content/nlp04_runtime_checkpoints')/run_id/'latest_checkpoint.pt']:
        if p.is_file():
            size=p.stat().st_size
            p.unlink()
            removed.append({'path':str(p),'size_bytes':size})
    save_json_atomic(cleanup_path,{'run_id':run_id,'removed_recovery_checkpoints':removed,
        'removed_bytes':sum(x['size_bytes'] for x in removed),'best_model_retained':True,
        'retained_model_path':rel(model_path),'official_test_used':False})
cleanup=read_json(cleanup_path)
assert model_path.is_dir()
assert calculate_sha256(DECISION_LOG_PATH)==decision_hash_before
print('\nFull-data RoBERTa validation results')
display(pd.DataFrame([optimized_metrics]).round(6))
display(optimized_per_label.round(6))
print('\nScaling comparison')
display(scaling_df.round(6))
display(per_label_scaling.round(6))
print('\nSelected thresholds')
display(threshold_df.round(6))
print('\nS02 final summary')
for key,value in summary.items(): print(key,':',value)
print('Verified retained artifacts:',len(records_df))
print('Registered experiments:',len(registry_after))
print('Recovery storage released (GiB):',round(cleanup['removed_bytes']/1024**3,3))
print('Decision log: Unchanged')
print('Additional model training performed: No')
print('Official test dataset: Not loaded')
print('Final project model frozen: No')
print('Cell 8 completed successfully.')
