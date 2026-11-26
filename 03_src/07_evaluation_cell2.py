# Notebook 07, Cell 2: frozen classical model inference
import gc
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 first.'
assert len(CLASSICAL_RUN_IDS) == 12
assert len(test_df) == EXPECTED_TEST_SAMPLES == 63978
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
np.testing.assert_array_equal(test_df['id'].astype(str).to_numpy(), test_ids)
np.testing.assert_array_equal(test_df[LABELS].to_numpy(), y_test)

# A small validation replay checks the original feature construction and score method.
validation_path2 = PROJECT_ROOT / '01_data/processed/validation.parquet'
assert calculate_sha256(validation_path2) == frozen_manifest['validation_dataset_sha256']
val2 = pd.read_parquet(validation_path2, columns=['id', 'comment_text', *LABELS])
assert len(val2) == 31911 and val2['id'].is_unique
rng2 = np.random.default_rng(SEED)
idx2 = list(rng2.choice(len(val2), size=192, replace=False))
for label in LABELS:
    positive = np.flatnonzero(val2[label].to_numpy() == 1)
    idx2.extend(rng2.choice(positive, size=min(8, len(positive)), replace=False))
val_sample2 = val2.iloc[np.unique(idx2)].copy().reset_index(drop=True)
del val2


def load_validation_scores2(entry):
    path = verify_frozen_file(entry['validation_predictions'])
    if path.suffix.lower() == '.parquet':
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == '.csv':
        frame = pd.read_csv(path, dtype={'id': str})
    else:
        raise ValueError(f'Unsupported validation score file: {path}')
    frame['id'] = frame['id'].astype(str)
    assert frame['id'].is_unique and len(frame) == 31911
    if 'run_id' in frame:
        assert frame['run_id'].eq(entry['run_id']).all()
    score_columns = [f'score_{l}' for l in LABELS]
    true_columns = [f'true_{l}' for l in LABELS]
    if not set(score_columns).issubset(frame.columns):
        score_columns = [f'{l}_score' for l in LABELS]
    if not set(true_columns).issubset(frame.columns):
        true_columns = [f'{l}_true' for l in LABELS]
    sample = frame.set_index('id').loc[val_sample2['id'].astype(str)]
    np.testing.assert_array_equal(sample[true_columns].to_numpy(), val_sample2[LABELS].to_numpy())
    return sample[score_columns].to_numpy(dtype=np.float64)


def feature_key2(entry):
    return (entry['feature_mode'], tuple((v['role'], v['path'], v['sha256']) for v in entry['vectorizers']),
            bool(entry['combined_final_l2_normalization']))


def build_features2(vectorizers, entry, texts):
    mode = entry['feature_mode']
    if mode == 'combined':
        assert entry['combined_feature_order'] == ['word', 'character']
        assert entry['combined_final_l2_normalization'] is True
        word = vectorizers['word'].transform(texts)
        char = vectorizers['character'].transform(texts)
        matrix = sp.hstack([word, char], format='csr')
        del word, char
        matrix = normalize(matrix, norm='l2', axis=1, copy=False)
    else:
        assert mode in ('word', 'character')
        matrix = vectorizers[mode].transform(texts).tocsr()
    assert sp.issparse(matrix) and matrix.shape[0] == len(texts)
    assert np.isfinite(matrix.data).all()
    return matrix


def infer2(model, matrix, entry):
    method = entry['inference_method']
    assert method in ('predict_proba', 'decision_function')
    result = np.asarray(getattr(model, method)(matrix), dtype=np.float64)
    assert result.shape == (matrix.shape[0], len(LABELS)), (entry['run_id'], result.shape)
    assert np.isfinite(result).all()
    if method == 'predict_proba':
        assert (result >= 0).all() and (result <= 1).all()
    return result


def run_paths2(rid):
    root = PREDICTIONS_DIR / rid
    root.mkdir(parents=True, exist_ok=True)
    return root / 'test_predictions.parquet', root / 'inference_summary.json'


def check_prediction2(path, entry):
    frame = pd.read_parquet(path)
    assert len(frame) == EXPECTED_TEST_SAMPLES
    assert frame['id'].is_unique and frame['run_id'].eq(entry['run_id']).all()
    assert frame['split'].eq('test_official').all()
    np.testing.assert_array_equal(frame['id'].astype(str).to_numpy(), test_ids)
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
    scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    assert np.isfinite(scores).all()
    if entry['inference_method'] == 'predict_proba':
        assert (scores >= 0).all() and (scores <= 1).all()
    for name, thresholds in [('default', entry['default_threshold']),
                             ('optimized', [entry['optimized_thresholds'][l] for l in LABELS])]:
        np.testing.assert_array_equal(frame[[f'pred_{name}_{l}' for l in LABELS]].to_numpy(),
                                      (scores >= np.asarray(thresholds)).astype(np.int8))
    return scores


def completed2(entry):
    pred_path, summary_path = run_paths2(entry['run_id'])
    if not summary_path.exists():
        return None
    record = read_json(summary_path)
    assert record['run_id'] == entry['run_id']
    assert record['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
    assert record['test_dataset_sha256'] == EXPECTED_TEST_HASH
    assert record['predictions_path'] == str(pred_path.relative_to(PROJECT_ROOT))
    assert record['predictions_sha256'] == calculate_sha256(pred_path)
    assert record['test_samples'] == EXPECTED_TEST_SAMPLES
    assert record['model_training_performed'] is False and record['thresholds_changed'] is False
    check_prediction2(pred_path, entry)
    return record


# Group models by their exact frozen vectorizer dependencies.
groups2 = defaultdict(list)
records2 = {}
for rid in CLASSICAL_RUN_IDS:
    entry = FROZEN_MODELS[rid]
    assert entry['kind'] == 'classical' and entry['labels'] == LABELS
    prior = completed2(entry)
    if prior is not None:
        records2[rid] = prior
        print('Verified existing predictions:', rid, flush=True)
    else:
        groups2[feature_key2(entry)].append(entry)

print('Pending feature groups:', len(groups2), flush=True)
for entries in groups2.values():
    reference = entries[0]
    group_start = time.perf_counter()
    vectorizers = {rec['role']: joblib.load(verify_frozen_file(rec)) for rec in reference['vectorizers']}
    validation_matrix = build_features2(vectorizers, reference, val_sample2['comment_text'].tolist())
    print('\nFeature group:', reference['feature_mode'], [e['run_id'] for e in entries], flush=True)

    # Check all models on validation before scoring the official test for this feature group.
    validation_errors = {}
    load_warnings = {}
    for entry in entries:
        rid = entry['run_id']
        cfg = read_json(verify_frozen_file(entry['config']))
        verify_frozen_file(entry['threshold_file'])
        assert len(entry['model_files']) == 1
        model_path = verify_frozen_file(entry['model_files'][0])
        expected = load_validation_scores2(entry)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = joblib.load(model_path)
        load_warnings[rid] = [str(w.message) for w in captured]
        observed = infer2(model, validation_matrix, entry)
        error = float(np.max(np.abs(observed - expected)))
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=5e-6, err_msg=rid)
        validation_errors[rid] = error
        print('Validation replay:', rid, 'max score error:', error, flush=True)
        del model, observed, expected
        gc.collect()
    del validation_matrix

    # Build one sparse test matrix and reuse it for all matching models.
    start = time.perf_counter()
    matrix = build_features2(vectorizers, reference, test_df['comment_text'].tolist())
    feature_seconds = time.perf_counter() - start
    print('Test matrix:', matrix.shape, 'nonzero:', matrix.nnz, 'seconds:', round(feature_seconds, 2), flush=True)
    del vectorizers
    gc.collect()
    for entry in entries:
        rid = entry['run_id']
        pred_path, summary_path = run_paths2(rid)
        assert not summary_path.exists()
        model = joblib.load(PROJECT_ROOT / entry['model_files'][0]['path'])
        started = time.perf_counter()
        scores = np.empty((EXPECTED_TEST_SAMPLES, len(LABELS)), dtype=np.float64)
        for start in range(0, EXPECTED_TEST_SAMPLES, 2048):
            end = min(start + 2048, EXPECTED_TEST_SAMPLES)
            scores[start:end] = infer2(model, matrix[start:end], entry)
        seconds = time.perf_counter() - started
        del model
        default = (scores >= entry['default_threshold']).astype(np.int8)
        cutoffs = np.array([entry['optimized_thresholds'][l] for l in LABELS])
        optimized = (scores >= cutoffs).astype(np.int8)
        frame = pd.DataFrame({'id': test_ids, 'split': 'test_official', 'run_id': rid})
        for j, label in enumerate(LABELS):
            frame[f'true_{label}'] = y_test[:, j]
            frame[f'score_{label}'] = scores[:, j]
            frame[f'pred_default_{label}'] = default[:, j]
            frame[f'pred_optimized_{label}'] = optimized[:, j]
        temp_path = pred_path.with_suffix('.tmp.parquet')
        frame.to_parquet(temp_path, index=False, compression='zstd')
        np.testing.assert_array_equal(check_prediction2(temp_path, entry), scores)
        temp_path.replace(pred_path)
        record = {'run_id': rid, 'kind': 'classical', 'score_type': entry['score_type'],
                  'test_samples': EXPECTED_TEST_SAMPLES, 'test_dataset_sha256': EXPECTED_TEST_HASH,
                  'frozen_manifest_sha256': EXPECTED_FROZEN_HASH,
                  'predictions_path': str(pred_path.relative_to(PROJECT_ROOT)),
                  'predictions_sha256': calculate_sha256(pred_path),
                  'validation_replay_samples': len(val_sample2),
                  'validation_replay_max_score_error': validation_errors[rid],
                  'feature_transformation_seconds_shared_group': feature_seconds,
                  'feature_group_runs': [e['run_id'] for e in entries],
                  'feature_columns': int(matrix.shape[1]), 'inference_seconds': seconds,
                  'model_size_bytes': entry['model_files'][0]['size_bytes'],
                  'device': 'cpu', 'sklearn_version': sklearn.__version__,
                  'model_load_warnings': load_warnings[rid], 'model_training_performed': False,
                  'thresholds_changed': False}
        save_json_atomic(summary_path, record)
        records2[rid] = record
        print('Saved:', rid, 'inference seconds:', round(seconds, 2), flush=True)
        del scores, default, optimized, frame
        gc.collect()
    del matrix
    gc.collect()
    print('Feature group finished in seconds:', round(time.perf_counter() - group_start, 2), flush=True)

assert set(records2) == set(CLASSICAL_RUN_IDS)
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
classical_records2 = [records2[rid] for rid in CLASSICAL_RUN_IDS]
save_json_atomic(LOGS_DIR / 'classical_test_inference_summary.json', {
    'status': 'Completed', 'frozen_manifest_sha256': EXPECTED_FROZEN_HASH,
    'test_dataset_sha256': EXPECTED_TEST_HASH, 'completed_runs': len(classical_records2),
    'test_samples_per_run': EXPECTED_TEST_SAMPLES, 'runs': classical_records2,
    'model_training_performed': False, 'thresholds_changed': False})
summary_df2 = pd.DataFrame([{'Run ID': rec['run_id'], 'Test Samples': rec['test_samples'],
    'Validation Replay Max Score Error': rec['validation_replay_max_score_error'],
    'Inference Seconds': rec['inference_seconds'], 'Prediction SHA256': rec['predictions_sha256']}
    for rec in classical_records2])
save_csv_atomic(TABLES_DIR / 'classical_inference_summary.csv', summary_df2)
display(summary_df2.drop(columns='Prediction SHA256'))
print('\nClassical configurations completed: 12')
print('Official test samples per run: 63978')
print('Prediction artifacts verified: 12')
print('GPU inference and model training: None')
print('Frozen thresholds and manifest: Unchanged')
print('Cell 2 completed successfully.')
