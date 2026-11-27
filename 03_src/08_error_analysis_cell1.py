# Notebook 08, Cell 1: verify the completed evaluation and prepare error analysis.
from pathlib import Path
import json, hashlib, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path('/content/drive/MyDrive/NLP04_Toxic_Comment_Classification')
if not PROJECT_ROOT.is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
assert PROJECT_ROOT.is_dir()
CONFIG_DIR = PROJECT_ROOT/'00_config'
SRC_DIR = PROJECT_ROOT/'03_src'
PROCESSED_DIR = PROJECT_ROOT/'01_data/processed'
SPLITS_DIR = PROJECT_ROOT/'01_data/splits'
NOTEBOOK_DIR = PROJECT_ROOT/'02_notebooks/08_error_analysis'
RESULTS_DIR = NOTEBOOK_DIR/'results'
TABLES_DIR = RESULTS_DIR/'tables'
FIGURES_DIR = RESULTS_DIR/'figures'
LOGS_DIR = RESULTS_DIR/'logs'
RUN_CONFIGS_DIR = RESULTS_DIR/'configs'
PREDICTIONS_DIR = RESULTS_DIR/'predictions'
PRIVATE_DIR = RESULTS_DIR/'private_review'
for path in [TABLES_DIR, FIGURES_DIR, LOGS_DIR, RUN_CONFIGS_DIR, PREDICTIONS_DIR, PRIVATE_DIR]:
    path.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = CONFIG_DIR/'experiment_registry.csv'
DECISION_LOG_PATH = CONFIG_DIR/'project_decisions.csv'
EVALUATION_RESULTS_DIR = PROJECT_ROOT/'02_notebooks/07_final_evaluation/results'
EVALUATION_COMPLETION_PATH = EVALUATION_RESULTS_DIR/'logs/evaluation_completion_summary.json'
FROZEN_MANIFEST_PATH = PROJECT_ROOT/'02_notebooks/06_model_optimization/results/configs/frozen_manifest.json'
EXPECTED_FROZEN_HASH = '229adb5d6166a626f6b7f7fddecf38b4de58d21e56ec4367465ad29174ef2fd1'
EXPECTED_TEST_HASH = 'cddc70d876b48b9fe3d5db8109a740430a7f4d9f50f9fcd77f4c7d566035f328'
EXPECTED_INDEX_HASH = 'c7096a87e7865508c71f17dd5a33bcb5bf4c13b077568f9555aafdb0d0a00213'
SEED = 42

def read_json(path):
    return json.loads(Path(path).read_text())

def calculate_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json_atomic(path, content):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(content, indent=4, allow_nan=False))
    assert read_json(tmp) == content
    tmp.replace(path)

def save_csv_atomic(path, frame):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)

def verify_file(record):
    path = (PROJECT_ROOT/record['path']).resolve()
    assert path.is_relative_to(PROJECT_ROOT.resolve()) and path.is_file(), path
    assert path.stat().st_size == record['size_bytes'], path
    assert calculate_sha256(path) == record['sha256'], path
    return path

# Verify the immutable protocol and completed evaluation.
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
frozen_manifest = read_json(FROZEN_MANIFEST_PATH)
evaluation_completion = read_json(EVALUATION_COMPLETION_PATH)
assert evaluation_completion['status'] == 'Completed'
assert evaluation_completion['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
assert evaluation_completion['test_dataset_sha256'] == EXPECTED_TEST_HASH
assert evaluation_completion['prediction_index_sha256'] == EXPECTED_INDEX_HASH
assert evaluation_completion['official_test_evaluated']
assert evaluation_completion['evaluated_configurations'] == 19
assert evaluation_completion['final_project_model_frozen']
assert not evaluation_completion['thresholds_changed']
assert not evaluation_completion['test_driven_selection_performed']
assert frozen_manifest['thresholds_frozen'] and frozen_manifest['model_parameters_frozen']
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
LABELS = frozen_manifest['labels']
assert LABELS == ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
PRIMARY_RUN_ID = frozen_manifest['primary_selected_run_id']
BASELINE_RUN_ID = frozen_manifest['selected_models']['best_classical']
assert PRIMARY_RUN_ID == 'S02_ROBERTA_FOCAL_FULL_S42'
assert BASELINE_RUN_ID == 'S01_LR_COMBINED_BALANCED_FULL_S42'
FROZEN_MODELS = {e['run_id']:e for e in frozen_manifest['models']}
assert len(FROZEN_MODELS) == 19
assert evaluation_completion['primary_selected_run_id'] == PRIMARY_RUN_ID
inventory = {r['path']:r for r in evaluation_completion['artifact_inventory']}
assert len(inventory) == len(evaluation_completion['artifact_inventory']) == 79

def verify_evaluation_artifact(path):
    return verify_file(inventory[str(Path(path).relative_to(PROJECT_ROOT))])

print('Completed evaluation and frozen protocol verified.')

# Restore test and validation data for descriptive comparison only.
TEST_PATH = PROJECT_ROOT/evaluation_completion['test_dataset_path']
VALIDATION_PATH = PROCESSED_DIR/'validation.parquet'
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
assert calculate_sha256(VALIDATION_PATH) == frozen_manifest['validation_dataset_sha256']
test_df = pd.read_parquet(TEST_PATH, columns=['id','comment_text',*LABELS])
validation_df = pd.read_parquet(VALIDATION_PATH, columns=['id','comment_text',*LABELS])
for frame, n in [(test_df,63978),(validation_df,31911)]:
    assert len(frame) == n and frame['id'].notna().all() and frame['id'].is_unique
    assert frame['comment_text'].notna().all()
    assert np.isin(frame[LABELS].to_numpy(), [0,1]).all()
assert set(test_df['id']).isdisjoint(validation_df['id'])
y_test = test_df[LABELS].to_numpy(dtype=np.int8)
y_validation = validation_df[LABELS].to_numpy(dtype=np.int8)
test_ids = test_df['id'].astype(str).to_numpy()

index_path = PROJECT_ROOT/evaluation_completion['prediction_index_path']
assert calculate_sha256(index_path) == EXPECTED_INDEX_HASH
verify_evaluation_artifact(index_path)
prediction_index = pd.read_csv(index_path).set_index('Run ID')
assert prediction_index.index.is_unique and set(prediction_index.index) == set(FROZEN_MODELS)
per_label_path = verify_evaluation_artifact(EVALUATION_RESULTS_DIR/'tables/test_per_label_metrics.csv')
per_label_test = pd.read_csv(per_label_path)
assert len(per_label_test) == 228

# Verify the selected models' saved predictions, not the model weights.
verified_sources = []
def load_test_predictions(rid):
    row = prediction_index.loc[rid]
    path = PROJECT_ROOT/row['Prediction Path']
    verify_evaluation_artifact(path)
    assert inventory[row['Prediction Path']]['sha256'] == row['SHA256']
    assert int(row['Rows']) == len(test_df)
    entry = FROZEN_MODELS[rid]
    assert row['Score Type'] == entry['score_type'] and entry['comparison_operator'] == '>='
    frame = pd.read_parquet(path)
    assert len(frame) == len(test_df) and frame['id'].is_unique
    assert frame['run_id'].eq(rid).all() and frame['split'].eq('test_official').all()
    np.testing.assert_array_equal(frame['id'].astype(str).to_numpy(), test_ids)
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
    scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=float)
    assert np.isfinite(scores).all() and ((scores >= 0)&(scores <= 1)).all()
    cutoff = np.array([entry['optimized_thresholds'][l] for l in LABELS])
    pred = scores >= cutoff
    np.testing.assert_array_equal(frame[[f'pred_optimized_{l}' for l in LABELS]].to_numpy(), pred)
    per = per_label_test[(per_label_test['Run ID']==rid)&(per_label_test['Operating Point']=='optimized')].set_index('Label').loc[LABELS]
    for name, counts in [('TP',(pred&(y_test==1)).sum(0)),('FP',(pred&(y_test==0)).sum(0)),('FN',((~pred)&(y_test==1)).sum(0)),('TN',((~pred)&(y_test==0)).sum(0))]:
        np.testing.assert_array_equal(per[name].to_numpy(),counts)
    np.testing.assert_allclose(per['Average Precision'].to_numpy(),average_precision_score(y_test,scores,average=None),rtol=0,atol=1e-12)
    verified_sources.append({'run_id':rid,'path':row['Prediction Path'],'sha256':row['SHA256']})
    print('Verified test scores and confusion counts:',rid)
    return frame

primary_predictions = load_test_predictions(PRIMARY_RUN_ID)
baseline_predictions = load_test_predictions(BASELINE_RUN_ID)
entry = FROZEN_MODELS[PRIMARY_RUN_ID]
val_path = verify_file(entry['validation_predictions'])
primary_validation_predictions = pd.read_parquet(val_path)
assert len(primary_validation_predictions) == len(validation_df)
assert primary_validation_predictions['id'].is_unique
np.testing.assert_array_equal(primary_validation_predictions['id'].astype(str).to_numpy(),validation_df['id'].astype(str).to_numpy())
np.testing.assert_array_equal(primary_validation_predictions[[f'true_{l}' for l in LABELS]].to_numpy(),y_validation)
val_scores = primary_validation_predictions[[f'score_{l}' for l in LABELS]].to_numpy(dtype=float)
assert np.isfinite(val_scores).all() and ((val_scores>=0)&(val_scores<=1)).all()
np.testing.assert_allclose(average_precision_score(y_validation,val_scores,average='macro'),entry['validation_macro_ap'],rtol=0,atol=1e-12)
np.testing.assert_array_equal(primary_validation_predictions[[f'pred_optimized_{l}' for l in LABELS]].to_numpy(),val_scores>=np.array([entry['optimized_thresholds'][l] for l in LABELS]))
print('Verified primary validation predictions.')

# Record a bounded review plan; severe toxicity added due to observed error burden.
review_quotas = {'toxic':{'FP':40,'FN':40},'severe_toxic':{'FP':10,'FN':10},'threat':{'FP':20,'FN':20},'identity_hate':{'FP':20,'FN':20}}
assert sum(sum(v.values()) for v in review_quotas.values()) == 180
error_analysis_setup = {
    'notebook':'08_error_analysis','frozen_manifest_sha256':EXPECTED_FROZEN_HASH,
    'evaluation_completion_sha256':calculate_sha256(EVALUATION_COMPLETION_PATH),
    'test_dataset_sha256':EXPECTED_TEST_HASH,'validation_dataset_sha256':frozen_manifest['validation_dataset_sha256'],
    'prediction_index_sha256':EXPECTED_INDEX_HASH,'primary_run_id':PRIMARY_RUN_ID,'baseline_run_id':BASELINE_RUN_ID,
    'test_samples':len(test_df),'validation_samples':len(validation_df),'labels':LABELS,'seed':SEED,
    'test_prediction_sources':verified_sources,'validation_prediction_source':entry['validation_predictions'],
    'manual_review_quotas':review_quotas,'manual_review_unit':'label-level case',
    'manual_review_sampling':'uniform without replacement within label and error outcome; all available if below quota',
    'scope_revision':'Reduce toxic quotas from 50 to 40 per outcome and add 10 severe-toxic cases per outcome; retain planned rare-label strata and total 180. Descriptive post-test audit only.',
    'thresholds_changed':False,'test_driven_selection_performed':False,'model_training_performed':False,
    'additional_inference_performed':False,'raw_comment_review_private':True}
setup_path = RUN_CONFIGS_DIR/'error_analysis_setup.json'
if setup_path.exists():
    assert read_json(setup_path) == error_analysis_setup, 'Existing setup differs; inspect before overwriting.'
else:
    save_json_atomic(setup_path,error_analysis_setup)
print('\nLabel support for descriptive validation/test comparison')
display(pd.DataFrame({'Label':LABELS,'Validation Positives':y_validation.sum(0),'Test Positives':y_test.sum(0),'Validation Prevalence (%)':y_validation.mean(0)*100,'Test Prevalence (%)':y_test.mean(0)*100}).round(4))
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
print('\nError analysis setup summary')
print('Primary model:',PRIMARY_RUN_ID)
print('Classical comparison:',BASELINE_RUN_ID)
print('Verified test prediction files: 2')
print('Verified primary validation prediction file: 1')
print('Test comments:',len(test_df))
print('Validation comments:',len(validation_df))
print('Planned manual label-level cases: 180')
print('Model training and inference: None')
print('Frozen protocol: Unchanged')
print('Cell 1 completed successfully.')
