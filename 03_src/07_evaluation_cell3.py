# Notebook 07, Cell 3: prepare reusable frozen transformer test inputs
import gc
import time
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers
from transformers import AutoTokenizer

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 first.'
assert len(TRANSFORMER_RUN_IDS) == 7
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
assert len(test_df) == EXPECTED_TEST_SAMPLES == 63978
np.testing.assert_array_equal(test_df['id'].astype(str).to_numpy(), test_ids)

# Confirm classical inference finished before preparing the transformers.
classical3 = read_json(LOGS_DIR / 'classical_test_inference_summary.json')
assert classical3['status'] == 'Completed'
assert classical3['completed_runs'] == 12
assert classical3['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
assert classical3['test_dataset_sha256'] == EXPECTED_TEST_HASH
assert set(r['run_id'] for r in classical3['runs']) == set(CLASSICAL_RUN_IDS)
for rec in classical3['runs']:
    assert calculate_sha256(PROJECT_ROOT / rec['predictions_path']) == rec['predictions_sha256']

CACHE_DIR3 = TOKENIZED_DIR / 'official_test_frozen'
CACHE_DIR3.mkdir(parents=True, exist_ok=True)
TOKENIZATION_CONFIG_PATH = RUN_CONFIGS_DIR / 'transformer_test_tokenization.json'
CACHE_SCHEMA3 = pa.schema([
    ('id', pa.string()), ('row_index', pa.int64()),
    ('input_ids', pa.list_(pa.int32()))
])

# Group models by exact frozen tokenizer files and input settings.
groups3 = defaultdict(list)
specs3 = {}
for rid in TRANSFORMER_RUN_IDS:
    entry = FROZEN_MODELS[rid]
    model_dir = PROJECT_ROOT / entry['model_path']
    # All non-weight files in the frozen checkpoint are small configuration files.
    tokenizer_files = []
    for rec in entry['model_files']:
        name = Path(rec['path']).name
        if name.endswith(('.safetensors', '.bin', '.pt', '.pth')) or name == 'config.json':
            continue
        path = verify_frozen_file(rec)
        assert path.parent == model_dir.resolve(), (rid, path)
        tokenizer_files.append({'filename': name, 'sha256': rec['sha256']})
    assert any(r['filename'] == 'tokenizer.json' for r in tokenizer_files), rid
    assert any(r['filename'] == 'tokenizer_config.json' for r in tokenizer_files), rid
    spec = {
        'checkpoint': entry['checkpoint'], 'revision': entry['revision'],
        'maximum_length': int(entry['maximum_length']),
        'padding_multiple': int(entry['padding_multiple']),
        'tokenizer_files': sorted(tokenizer_files, key=lambda r: r['filename']),
        'test_dataset_sha256': EXPECTED_TEST_HASH,
        'test_samples': EXPECTED_TEST_SAMPLES,
        'truncation_side': 'right', 'padding_side': 'right',
        'padding': False, 'add_special_tokens': True,
        'transformers_version': transformers.__version__,
        'tokenizers_version': tokenizers.__version__
    }
    assert spec['maximum_length'] == 256 and spec['padding_multiple'] == 8
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    groups3[key].append(rid)
    specs3[key] = spec
print('Exact tokenizer groups:', len(groups3), flush=True)


def tokenize_texts3(tokenizer, texts, spec):
    return tokenizer(list(texts), truncation=True, max_length=spec['maximum_length'],
                     padding=False, add_special_tokens=True,
                     return_attention_mask=False, return_token_type_ids=False)['input_ids']


def verify_cache3(path, tokenizer, spec):
    table = pq.read_table(path)
    assert table.schema.equals(CACHE_SCHEMA3, check_metadata=False)
    assert table.num_rows == EXPECTED_TEST_SAMPLES
    np.testing.assert_array_equal(np.asarray(table['id'].to_pylist(), dtype=str), test_ids)
    np.testing.assert_array_equal(table['row_index'].to_numpy(), np.arange(EXPECTED_TEST_SAMPLES))
    arr = table['input_ids'].combine_chunks()
    lengths = np.asarray(arr.value_lengths().to_numpy(), dtype=np.int32)
    assert arr.null_count == 0 and arr.values.null_count == 0
    assert lengths.min() >= 2 and lengths.max() <= spec['maximum_length']
    flat = arr.values.to_numpy()
    assert flat.min() >= 0 and flat.max() < len(tokenizer)
    sample = np.sort(np.random.default_rng(SEED).choice(EXPECTED_TEST_SAMPLES, 256, replace=False))
    fresh = tokenize_texts3(tokenizer, test_df.iloc[sample]['comment_text'], spec)
    assert fresh == table.take(pa.array(sample))['input_ids'].to_pylist()
    return {'min_length': int(lengths.min()), 'mean_length': float(lengths.mean()),
            'median_length': float(np.median(lengths)), 'p95_length': float(np.percentile(lengths, 95)),
            'max_length': int(lengths.max()), 'sequences_at_limit': int((lengths == spec['maximum_length']).sum()),
            'tokenization_replay_samples': len(sample)}

records3 = []
run_cache_map3 = {}
for key, runs in groups3.items():
    spec = specs3[key]
    entry = FROZEN_MODELS[runs[0]]
    model_dir = PROJECT_ROOT / entry['model_path']
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    assert tok.is_fast and tok.padding_side == tok.truncation_side == 'right'
    backbone = 'roberta' if 'roberta' in spec['checkpoint'].lower() else 'distilbert'
    if backbone == 'roberta':
        assert tok.vocab_size == 50265 and tok.pad_token_id == 1
        assert tok.bos_token_id == 0 and tok.eos_token_id == 2
    else:
        assert 'distilbert' in spec['checkpoint'].lower()
        assert tok.vocab_size == 30522 and tok.pad_token_id == 0
        assert tok.cls_token_id == 101 and tok.sep_token_id == 102
    path = CACHE_DIR3 / f'{backbone}_test_{key[:16]}.parquet'
    manifest_path = path.with_suffix('.json')
    print('\nTokenizer:', backbone, '| runs:', runs, flush=True)
    if manifest_path.exists():
        prior = read_json(manifest_path)
        assert prior['specification'] == spec
        assert path.is_file() and calculate_sha256(path) == prior['cache_sha256']
        tokenization_seconds = prior['tokenization_seconds']
        print('Existing verified cache reused.', flush=True)
    else:
        # If interrupted before the manifest was written, verify the orphaned cache.
        if path.exists():
            verify_cache3(path, tok, spec)
            tokenization_seconds = None
            print('Existing cache recovered and verified.', flush=True)
        else:
            start_time = time.perf_counter()
            tmp = path.with_suffix('.tmp.parquet')
            with pq.ParquetWriter(tmp, CACHE_SCHEMA3, compression='zstd') as writer:
                for start in range(0, EXPECTED_TEST_SAMPLES, 1024):
                    end = min(start + 1024, EXPECTED_TEST_SAMPLES)
                    ids = tokenize_texts3(tok, test_df.iloc[start:end]['comment_text'], spec)
                    assert len(ids) == end-start and all(2 <= len(x) <= 256 for x in ids)
                    writer.write_table(pa.Table.from_pydict({
                        'id': test_ids[start:end].tolist(), 'row_index': list(range(start,end)),
                        'input_ids': ids}, schema=CACHE_SCHEMA3))
                    if end % 20480 == 0 or end == EXPECTED_TEST_SAMPLES:
                        print(f'Tokenized {end:,}/{EXPECTED_TEST_SAMPLES:,}', flush=True)
            verify_cache3(tmp, tok, spec)
            tmp.replace(path)
            tokenization_seconds = time.perf_counter() - start_time
        prior = {'specification': spec, 'cache_sha256': calculate_sha256(path),
                 'cache_size_bytes': path.stat().st_size,
                 'tokenization_seconds': tokenization_seconds}
        save_json_atomic(manifest_path, prior)
    stats = verify_cache3(path, tok, spec)
    rec = {'cache_key': key, 'backbone': backbone, 'run_ids': runs,
           'tokenizer_source_run_id': runs[0], 'cache_path': str(path.relative_to(PROJECT_ROOT)),
           'cache_sha256': prior['cache_sha256'], 'cache_manifest_path': str(manifest_path.relative_to(PROJECT_ROOT)),
           'cache_manifest_sha256': calculate_sha256(manifest_path),
           'cache_size_bytes': path.stat().st_size, 'tokenization_seconds': tokenization_seconds,
           'maximum_length': spec['maximum_length'], 'padding_multiple': spec['padding_multiple'],
           'pad_token_id': int(tok.pad_token_id), **stats}
    records3.append(rec)
    for rid in runs:
        run_cache_map3[rid] = key
    print('Cache verified:', path.name, '| MiB:', round(path.stat().st_size/1024**2, 3), flush=True)
    del tok
    gc.collect()

assert set(run_cache_map3) == set(TRANSFORMER_RUN_IDS)
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
summary3 = {'status': 'Completed', 'frozen_manifest_sha256': EXPECTED_FROZEN_HASH,
            'test_dataset_sha256': EXPECTED_TEST_HASH, 'test_samples': EXPECTED_TEST_SAMPLES,
            'tokenizer_groups': len(records3), 'transformer_runs': len(run_cache_map3),
            'run_cache_map': run_cache_map3, 'caches': records3,
            'model_training_performed': False, 'model_inference_performed': False,
            'thresholds_changed': False}
if TOKENIZATION_CONFIG_PATH.exists():
    assert read_json(TOKENIZATION_CONFIG_PATH) == summary3, 'Existing tokenization configuration differs.'
else:
    save_json_atomic(TOKENIZATION_CONFIG_PATH, summary3)
summary_table3 = pd.DataFrame([{'Backbone': r['backbone'], 'Runs': len(r['run_ids']),
    'Test Samples': EXPECTED_TEST_SAMPLES, 'Mean Length': r['mean_length'],
    'Median Length': r['median_length'], 'P95 Length': r['p95_length'],
    'Maximum Length': r['max_length'], 'Sequences At Limit': r['sequences_at_limit'],
    'Cache MiB': r['cache_size_bytes']/1024**2, 'Tokenization Seconds': r['tokenization_seconds'],
    'Replay Checks': r['tokenization_replay_samples']} for r in records3])
save_csv_atomic(TABLES_DIR / 'transformer_test_tokenization_summary.csv', summary_table3)
display(summary_table3.round(4))
print('\nTransformer configurations prepared:', len(run_cache_map3))
print('Verified shared tokenizer caches:', len(records3))
print('Official test comments per cache:', EXPECTED_TEST_SAMPLES)
print('Model training and inference: None')
print('Frozen thresholds and manifest: Unchanged')
print('Cell 3 completed successfully.')
