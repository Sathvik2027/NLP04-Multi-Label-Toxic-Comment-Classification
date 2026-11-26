# Notebook 07, Cell 4: frozen transformer inference
import os
import gc
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import transformers
import tokenizers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 after switching runtime.'
assert torch.cuda.is_available(), 'Select a GPU runtime and rerun Cell 1.'
assert len(TRANSFORMER_RUN_IDS) == 7
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
np.testing.assert_array_equal(test_df['id'].astype(str).to_numpy(), test_ids)
np.testing.assert_array_equal(test_df[LABELS].to_numpy(), y_test)
DEVICE4 = torch.device('cuda')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Retain original evaluation precision and batch size; document runtime differences.
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('Torch:', torch.__version__, '| Transformers:', transformers.__version__, flush=True)

classical4 = read_json(LOGS_DIR / 'classical_test_inference_summary.json')
assert classical4['status'] == 'Completed' and classical4['completed_runs'] == 12
assert classical4['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
assert classical4['test_dataset_sha256'] == EXPECTED_TEST_HASH
assert set(r['run_id'] for r in classical4['runs']) == set(CLASSICAL_RUN_IDS)
for rec in classical4['runs']:
    assert calculate_sha256(PROJECT_ROOT / rec['predictions_path']) == rec['predictions_sha256']

tokenization_path4 = RUN_CONFIGS_DIR / 'transformer_test_tokenization.json'
tokenization_hash4 = calculate_sha256(tokenization_path4)
tokenization4 = read_json(tokenization_path4)
assert tokenization4['status'] == 'Completed'
assert tokenization4['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
assert tokenization4['test_dataset_sha256'] == EXPECTED_TEST_HASH
assert tokenization4['test_samples'] == EXPECTED_TEST_SAMPLES
assert set(tokenization4['run_cache_map']) == set(TRANSFORMER_RUN_IDS)
cache_records4 = {r['cache_key']: r for r in tokenization4['caches']}
assert len(cache_records4) == tokenization4['tokenizer_groups']

validation_path4 = PROCESSED_DIR / 'validation.parquet'
assert calculate_sha256(validation_path4) == frozen_manifest['validation_dataset_sha256']
val4 = pd.read_parquet(validation_path4, columns=['id', 'comment_text', *LABELS])
assert len(val4) == 31911 and val4['id'].is_unique


def model_hash4(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def paths4(rid):
    root = PREDICTIONS_DIR / rid
    root.mkdir(parents=True, exist_ok=True)
    return root / 'test_predictions.parquet', root / 'inference_summary.json'


def check_predictions4(path, entry):
    frame = pd.read_parquet(path)
    assert len(frame) == EXPECTED_TEST_SAMPLES and frame['id'].is_unique
    assert frame['run_id'].eq(entry['run_id']).all()
    assert frame['split'].eq('test_official').all()
    np.testing.assert_array_equal(frame['id'].astype(str).to_numpy(), test_ids)
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
    scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    logits = frame[[f'logit_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    assert np.isfinite(scores).all() and np.isfinite(logits).all()
    assert (scores >= 0).all() and (scores <= 1).all()
    from scipy.special import expit
    np.testing.assert_allclose(scores, expit(logits), rtol=0, atol=2e-7)
    for name, threshold in [('default', entry['default_threshold']),
                            ('optimized', [entry['optimized_thresholds'][l] for l in LABELS])]:
        np.testing.assert_array_equal(frame[[f'pred_{name}_{l}' for l in LABELS]].to_numpy(),
                                      (scores >= np.asarray(threshold, dtype=np.float64)).astype(np.int8))
    return scores, logits


def completed4(entry):
    pred, summary = paths4(entry['run_id'])
    if not summary.exists():
        return None
    record = read_json(summary)
    assert record['run_id'] == entry['run_id'] and record['kind'] == 'transformer'
    assert record['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
    assert record['test_dataset_sha256'] == EXPECTED_TEST_HASH
    assert record['tokenization_configuration_sha256'] == tokenization_hash4
    assert record['model_state_sha256'] == entry['model_state_sha256']
    assert record['predictions_path'] == str(pred.relative_to(PROJECT_ROOT))
    assert record['predictions_sha256'] == calculate_sha256(pred)
    assert record['test_samples'] == EXPECTED_TEST_SAMPLES
    assert not record['model_training_performed'] and not record['thresholds_changed']
    check_predictions4(pred, entry)
    return record


def load_cache4(rec):
    path = PROJECT_ROOT / rec['cache_path']
    manifest_path = PROJECT_ROOT / rec['cache_manifest_path']
    assert path.stat().st_size == rec['cache_size_bytes']
    assert calculate_sha256(path) == rec['cache_sha256']
    assert calculate_sha256(manifest_path) == rec['cache_manifest_sha256']
    manifest = read_json(manifest_path)
    assert manifest['cache_sha256'] == rec['cache_sha256']
    assert manifest['specification']['test_dataset_sha256'] == EXPECTED_TEST_HASH
    table = pq.read_table(path)
    assert table.num_rows == EXPECTED_TEST_SAMPLES
    np.testing.assert_array_equal(np.asarray(table['id'].to_pylist(), dtype=str), test_ids)
    np.testing.assert_array_equal(table['row_index'].to_numpy(), np.arange(EXPECTED_TEST_SAMPLES))
    ids = table['input_ids'].to_pylist()
    assert all(2 <= len(x) <= rec['maximum_length'] for x in ids)
    return ids, manifest['specification']


@torch.inference_mode()
def forward4(model, sequences, tokenizer, pad_multiple):
    # Dynamic right padding matches the original validation procedure.
    features = [{'input_ids': x, 'attention_mask': [1]*len(x)} for x in sequences]
    batch = tokenizer.pad(features, padding=True, pad_to_multiple_of=pad_multiple, return_tensors='pt')
    assert batch['input_ids'].shape[1] <= 256
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(input_ids=batch['input_ids'].to(DEVICE4),
                       attention_mask=batch['attention_mask'].to(DEVICE4)).logits.float()
    scores = torch.sigmoid(logits)
    result = logits.cpu().numpy(), scores.cpu().numpy()
    assert result[0].shape == result[1].shape == (len(sequences), len(LABELS))
    assert np.isfinite(result[0]).all() and np.isfinite(result[1]).all()
    return result


def validation_replay4(model, tokenizer, entry, batch_size):
    # Replay whole original batches so padding and batch shape match training-time evaluation.
    rng = np.random.default_rng(SEED)
    starts = set(int(x)*batch_size for x in rng.choice(len(val4)//batch_size, 6, replace=False))
    for label in LABELS:
        positive = np.flatnonzero(val4[label].to_numpy() == 1)
        idx = int(rng.choice(positive))
        starts.add(idx//batch_size*batch_size)
    path = verify_frozen_file(entry['validation_predictions'])
    frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'id':str})
    assert len(frame) == len(val4) and frame['id'].is_unique
    if 'run_id' in frame:
        assert frame['run_id'].eq(entry['run_id']).all()
    frame['id'] = frame['id'].astype(str)
    frame = frame.set_index('id').loc[val4['id'].astype(str)]
    np.testing.assert_array_equal(frame[[f'true_{l}' for l in LABELS]].to_numpy(), val4[LABELS].to_numpy())
    expected_scores = frame[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
    logit_cols = [f'logit_{l}' for l in LABELS]
    expected_logits = frame[logit_cols].to_numpy(dtype=np.float64) if set(logit_cols).issubset(frame.columns) else None
    score_error = 0.0
    logit_error = 0.0 if expected_logits is not None else None
    count = 0
    for start in sorted(starts):
        end = min(start+batch_size, len(val4))
        ids = tokenizer(val4.iloc[start:end]['comment_text'].tolist(), truncation=True,
                        max_length=entry['maximum_length'], padding=False,
                        add_special_tokens=True, return_attention_mask=False,
                        return_token_type_ids=False)['input_ids']
        logits, scores = forward4(model, ids, tokenizer, entry['padding_multiple'])
        score_error = max(score_error, float(np.max(np.abs(scores-expected_scores[start:end]))))
        if expected_logits is not None:
            logit_error = max(logit_error, float(np.max(np.abs(logits-expected_logits[start:end]))))
        count += end-start
    # Cross-runtime guard is wider than the original same-runtime reload gate; record both.
    strict = score_error <= 5e-4 and (logit_error is None or logit_error <= 2e-3)
    assert score_error <= 2e-3 and (logit_error is None or logit_error <= 2e-2), (entry['run_id'], score_error, logit_error)
    return count, score_error, logit_error, strict


records4 = {}
current_cache_key = None
sequences4 = None
for rid in TRANSFORMER_RUN_IDS:
    entry = FROZEN_MODELS[rid]
    assert entry['kind'] == 'transformer' and entry['labels'] == LABELS
    assert entry['inference_precision'] == 'fp16_cuda' and entry['score_type'] == 'sigmoid_probability'
    prior = completed4(entry)
    if prior is not None:
        records4[rid] = prior
        print('Verified existing predictions:', rid, flush=True)
        continue
    key = tokenization4['run_cache_map'][rid]
    rec = cache_records4[key]
    assert rid in rec['run_ids']
    if current_cache_key != key:
        del sequences4
        sequences4, spec = load_cache4(rec)
        current_cache_key = key
    assert spec['checkpoint'] == entry['checkpoint'] and spec['revision'] == entry['revision']
    assert spec['maximum_length'] == entry['maximum_length'] and spec['padding_multiple'] == entry['padding_multiple']
    cfg = read_json(verify_frozen_file(entry['config']))
    verify_frozen_file(entry['threshold_file'])
    ckpt = read_json(verify_frozen_file(entry['checkpoint_manifest']))
    assert ckpt['model_state_sha256'] == entry['model_state_sha256']
    model_dir = PROJECT_ROOT / entry['model_path']
    for file_record in entry['model_files']:
        verify_frozen_file(file_record)
    actual_tokenizer_files = {Path(r['path']).name:r['sha256'] for r in entry['model_files']}
    assert all(actual_tokenizer_files[r['filename']] == r['sha256'] for r in spec['tokenizer_files'])
    batch_size = int(cfg['run_signature']['validation_batch_size'])
    assert batch_size == 32
    print('\nLoading:', rid, flush=True)
    tokenizer4 = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    assert tokenizer4.is_fast and tokenizer4.padding_side == tokenizer4.truncation_side == 'right'
    assert tokenizer4.pad_token_id == rec['pad_token_id']
    model4 = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    assert model4.config.num_labels == len(LABELS)
    assert model_hash4(model4) == entry['model_state_sha256'], rid
    model4 = model4.to(DEVICE4).eval()
    replay_count, replay_score_error, replay_logit_error, strict_replay = validation_replay4(model4, tokenizer4, entry, batch_size)
    print('Validation replay:', replay_count, '| score error:', replay_score_error, '| original strict gate:', strict_replay, flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    logits4 = np.empty((EXPECTED_TEST_SAMPLES, len(LABELS)), dtype=np.float32)
    scores4 = np.empty_like(logits4)
    for start in range(0, EXPECTED_TEST_SAMPLES, batch_size):
        end = min(start+batch_size, EXPECTED_TEST_SAMPLES)
        logits4[start:end], scores4[start:end] = forward4(model4, sequences4[start:end], tokenizer4, entry['padding_multiple'])
        if end % 10240 == 0 or end == EXPECTED_TEST_SAMPLES:
            print(f'{rid}: {end:,}/{EXPECTED_TEST_SAMPLES:,}', flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start_time
    peak_alloc = torch.cuda.max_memory_allocated()/1024**3
    peak_reserved = torch.cuda.max_memory_reserved()/1024**3
    del model4, tokenizer4
    gc.collect()
    torch.cuda.empty_cache()
    pred_path, summary_path = paths4(rid)
    scores64 = scores4.astype(np.float64)
    default = (scores64 >= entry['default_threshold']).astype(np.int8)
    optimized = (scores64 >= np.array([entry['optimized_thresholds'][l] for l in LABELS])).astype(np.int8)
    frame = pd.DataFrame({'id':test_ids, 'split':'test_official', 'run_id':rid})
    for j, label in enumerate(LABELS):
        frame[f'true_{label}'] = y_test[:,j]
        frame[f'logit_{label}'] = logits4[:,j]
        frame[f'score_{label}'] = scores4[:,j]
        frame[f'pred_default_{label}'] = default[:,j]
        frame[f'pred_optimized_{label}'] = optimized[:,j]
    tmp = pred_path.with_suffix('.tmp.parquet')
    frame.to_parquet(tmp, index=False, compression='zstd')
    saved_scores, saved_logits = check_predictions4(tmp, entry)
    np.testing.assert_array_equal(saved_scores, scores4)
    np.testing.assert_array_equal(saved_logits, logits4)
    tmp.replace(pred_path)
    record = {'run_id':rid, 'kind':'transformer', 'score_type':entry['score_type'],
              'test_samples':EXPECTED_TEST_SAMPLES, 'test_dataset_sha256':EXPECTED_TEST_HASH,
              'frozen_manifest_sha256':EXPECTED_FROZEN_HASH, 'tokenization_configuration_sha256':tokenization_hash4,
              'token_cache_sha256':rec['cache_sha256'], 'model_state_sha256':entry['model_state_sha256'],
              'predictions_path':str(pred_path.relative_to(PROJECT_ROOT)), 'predictions_sha256':calculate_sha256(pred_path),
              'validation_replay_samples':replay_count, 'validation_replay_max_score_error':replay_score_error,
              'validation_replay_max_logit_error':replay_logit_error, 'original_strict_replay_gate_passed':strict_replay,
              'cross_runtime_score_tolerance':0.002, 'cross_runtime_logit_tolerance':0.02,
              'inference_seconds':seconds, 'model_size_bytes':sum(r['size_bytes'] for r in entry['model_files'] if r['path'].endswith('.safetensors')),
              'peak_gpu_allocated_gib':peak_alloc, 'peak_gpu_reserved_gib':peak_reserved,
              'batch_size':batch_size, 'precision':'fp16_autocast_float32_sigmoid', 'device':'cuda',
              'gpu_name':torch.cuda.get_device_name(0), 'torch_version':str(torch.__version__),
              'transformers_version':transformers.__version__, 'tokenizers_version':tokenizers.__version__,
              'model_training_performed':False, 'thresholds_changed':False}
    save_json_atomic(summary_path, record)
    records4[rid] = record
    print('Saved:', rid, '| inference seconds:', round(seconds,2), flush=True)
    del frame, logits4, scores4, scores64, default, optimized, saved_scores, saved_logits
    gc.collect()

assert set(records4) == set(TRANSFORMER_RUN_IDS)
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(tokenization_path4) == tokenization_hash4
transformer_records4 = [records4[rid] for rid in TRANSFORMER_RUN_IDS]
save_json_atomic(LOGS_DIR / 'transformer_test_inference_summary.json', {
    'status':'Completed', 'frozen_manifest_sha256':EXPECTED_FROZEN_HASH,
    'test_dataset_sha256':EXPECTED_TEST_HASH, 'completed_runs':len(transformer_records4),
    'test_samples_per_run':EXPECTED_TEST_SAMPLES, 'runs':transformer_records4,
    'model_training_performed':False, 'thresholds_changed':False})
summary_df4 = pd.DataFrame([{'Run ID':r['run_id'], 'Test Samples':r['test_samples'],
                            'Replay Score Error':r['validation_replay_max_score_error'],
                            'Original Strict Replay':r['original_strict_replay_gate_passed'],
                            'Inference Seconds':r['inference_seconds'],
                            'Peak GPU Allocated GiB':r['peak_gpu_allocated_gib'],
                            'Prediction SHA256':r['predictions_sha256']} for r in transformer_records4])
save_csv_atomic(TABLES_DIR / 'transformer_inference_summary.csv', summary_df4)
display(summary_df4.drop(columns='Prediction SHA256'))
del sequences4, val4
gc.collect()
torch.cuda.empty_cache()
print('\nTransformer configurations completed: 7')
print('Official test samples per run:', EXPECTED_TEST_SAMPLES)
print('Prediction artifacts verified: 7')
print('Classical configurations previously completed: 12')
print('Model training: None')
print('Frozen thresholds and manifest: Unchanged')
print('Cell 4 completed successfully.')
