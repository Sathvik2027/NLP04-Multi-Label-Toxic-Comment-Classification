# Notebook 06, Cell 6: S02 full-data RoBERTa training pipeline
# Run after Notebook 06 Cells 1 and 5; no full experiment is trained here.
import os, sys, gc, math, time, copy, json, random, hashlib, shutil, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import transformers
import tokenizers
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

assert torch.cuda.is_available()
assert S02_RUN_ID == 'S02_ROBERTA_FOCAL_FULL_S42'
registry_hash_before = calculate_sha256(REGISTRY_PATH)
decision_hash_before = calculate_sha256(DECISION_LOG_PATH)
assert S02_RUN_ID not in pd.read_csv(REGISTRY_PATH)['run_id'].values

# Restore and verify the source artifacts from the saved configuration.
s02_token_cfg_path = RUN_CONFIGS_DIR / 's02_tokenization_configuration.json'
s02_token_cfg = read_json(s02_token_cfg_path)
assert not s02_token_cfg['official_test_used']
for p, expected in [(FULL_TRAIN_PATH, EXPECTED_FULL_TRAINING_HASH),
                    (VALIDATION_PATH, EXPECTED_VALIDATION_HASH),
                    (SRC_DIR / 'evaluation_utils.py', evaluation_hash)]:
    assert calculate_sha256(p) == expected
S02_TRAIN_CACHE_PATH = PROJECT_ROOT / s02_token_cfg['training_token_cache_path']
S02_VALIDATION_CACHE_PATH = PROJECT_ROOT / s02_token_cfg['validation_token_cache_path']
for p,k in [(S02_TRAIN_CACHE_PATH,'training_token_cache_sha256'),
            (S02_VALIDATION_CACHE_PATH,'validation_token_cache_sha256')]:
    assert calculate_sha256(p) == s02_token_cfg[k]
ref_row = pd.read_csv(REGISTRY_PATH).set_index('run_id').loc['T06_FOCAL_S42']
reference_config_path = PROJECT_ROOT / ref_row['config_path']
assert calculate_sha256(reference_config_path) == s02_token_cfg['reference_config_sha256']
S02_REFERENCE_CONFIG = read_json(reference_config_path)
ref_sig = S02_REFERENCE_CONFIG['run_signature']
assert ref_sig['labels'] == LABELS
assert ref_sig['seed'] == SEED == 42
assert ref_sig['loss_configuration']['loss_name'] == 'FOCAL'
assert ref_sig['validation_token_cache_sha256'] == s02_token_cfg['validation_token_cache_sha256']
assert ref_sig['evaluation_module_sha256'] == evaluation_hash

# Copy the exact previous training settings, changing only data and schedule.
MODEL_CHECKPOINT = ref_sig['checkpoint']
MODEL_REVISION = ref_sig['revision']
MODEL_NAME = 'RoBERTa'
TRAINING_SAMPLES = 127628
VALIDATION_SAMPLES = 31911
TRAIN_BATCH_SIZE = int(ref_sig['physical_batch_size'])
EVAL_BATCH_SIZE = int(ref_sig['validation_batch_size'])
EFFECTIVE_BATCH_SIZE = int(ref_sig['effective_batch_size'])
GRAD_ACCUMULATION_STEPS = int(ref_sig['gradient_accumulation_steps'])
LEARNING_RATE = float(ref_sig['learning_rate'])
WEIGHT_DECAY = float(ref_sig['weight_decay'])
MAX_EPOCHS = int(ref_sig['maximum_epochs'])
EARLY_STOPPING_PATIENCE = int(ref_sig['early_stopping_patience'])
MAX_GRAD_NORM = float(ref_sig['maximum_gradient_norm'])
AMP_INITIAL_SCALE = float(ref_sig['amp_initial_scale'])
USE_FP16 = bool(ref_sig['fp16'])
MAX_LENGTH = int(ref_sig['maximum_length'])
PAD_MULTIPLE = int(ref_sig['padding_multiple'])
CHECKPOINT_INTERVAL = 500
LOG_INTERVAL = 100
WARMUP_RATIO = 0.1
assert (TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, EFFECTIVE_BATCH_SIZE, GRAD_ACCUMULATION_STEPS) == (16,32,32,2)
assert (MAX_LENGTH, PAD_MULTIPLE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE) == (256,8,3,2)
assert LEARNING_RATE == 2e-5 and WEIGHT_DECAY == .01 and MAX_GRAD_NORM == 1
assert AMP_INITIAL_SCALE == 1024 and USE_FP16
assert ref_sig['optimizer_betas'] == [0.9,0.999] and ref_sig['optimizer_epsilon'] == 1e-8
assert ref_sig['warmup_steps'] == math.ceil(ref_sig['total_training_steps'] * WARMUP_RATIO)
UPDATES_PER_EPOCH = math.ceil(TRAINING_SAMPLES / EFFECTIVE_BATCH_SIZE)
TOTAL_TRAINING_STEPS = UPDATES_PER_EPOCH * MAX_EPOCHS
WARMUP_STEPS = math.ceil(TOTAL_TRAINING_STEPS * WARMUP_RATIO)
BATCHES_PER_EPOCH = math.ceil(TRAINING_SAMPLES / TRAIN_BATCH_SIZE)
FINAL_ACCUMULATION_SIZE = TRAINING_SAMPLES % EFFECTIVE_BATCH_SIZE or EFFECTIVE_BATCH_SIZE
assert (UPDATES_PER_EPOCH,TOTAL_TRAINING_STEPS,WARMUP_STEPS,FINAL_ACCUMULATION_SIZE) == (3989,11967,1197,12)
LOSS_CONFIGS = {S02_RUN_ID: copy.deepcopy(ref_sig['loss_configuration'])}
LOSS_CONFIG_HASH = ref_sig['loss_configuration_sha256']  # source loss-policy fingerprint
INITIAL_STATE_HASH = ref_sig['initial_model_sha256']
assert INITIAL_STATE_HASH == 'c0cd540a6683c8590d66ac4922ab0e37f0a9dbf6f177155e8e5be7c8dfd94db5'

# Match the determinism policy recorded for the prior RoBERTa pipeline.
previous_pipeline_path = PROJECT_ROOT / '02_notebooks/05_roberta/results/configs/roberta_training_pipeline.json'
previous_pipeline = read_json(previous_pipeline_path)
assert previous_pipeline['training_configuration_finalized']
assert previous_pipeline['checkpoint_resume_verified']
assert previous_pipeline['run_signatures']['T06_FOCAL_S42'] == ref_sig

policy = previous_pipeline.get('determinism_policy')
policy_source = 'previous_training_pipeline'

if policy is None:

    smoke_path = (
        previous_pipeline_path.parent.parent
        / 'logs/roberta_smoke_test_summary.json'
    )

    expected_hash = previous_pipeline.get(
        'smoke_summary_sha256'
    )

    if (
        smoke_path.is_file()
        and expected_hash
        and calculate_sha256(smoke_path) == expected_hash
    ):
        previous_smoke = read_json(smoke_path)
        policy = previous_smoke.get('determinism_policy')
        policy_source = 'verified_previous_smoke_summary'

if policy is None:

    policy = {
        'deterministic_algorithms': True,
        'warn_only': False,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,
        'cublas_workspace_config': ':4096:8',
        'attention_backend_policy': 'unchanged',
        'precision': 'fp16'
    }

    policy_source = 'S02_explicit_reproducibility_policy'

print('Determinism policy source:', policy_source)
print('Determinism policy:', policy)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = policy['cublas_workspace_config']
torch.use_deterministic_algorithms(policy['deterministic_algorithms'], warn_only=policy['warn_only'])
torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
print('Reference training settings verified; S02 determinism policy configured.')

# Load row-aligned tokens and targets, using the original raw-text token caches.
class ToxicCommentDataset(Dataset):
    def __init__(self, input_ids, targets):
        self.input_ids = input_ids
        self.targets = np.asarray(targets, dtype=np.float32)
        assert self.targets.shape == (len(input_ids), NUM_LABELS)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, index):
        ids = self.input_ids[index]
        return {'input_ids': ids, 'attention_mask': [1]*len(ids),
                'labels': torch.from_numpy(self.targets[index]), 'row_index': index}

def load_s02_dataset(cache_path, dataset_path, count):
    frame = pd.read_parquet(dataset_path, columns=['id', *LABELS])
    table = pq.read_table(cache_path, columns=['id','row_index','input_ids'])
    assert len(frame) == table.num_rows == count
    np.testing.assert_array_equal(np.asarray(table['id'].to_pylist(),dtype=str), frame['id'].astype(str).to_numpy())
    np.testing.assert_array_equal(table['row_index'].to_numpy(), np.arange(count))
    y = frame[LABELS].to_numpy(dtype=np.float32)
    assert np.isin(y,[0,1]).all()
    assert ((y.sum(axis=0)>0) & (y.sum(axis=0)<count)).all()
    ids = table['input_ids'].to_pylist()
    lengths = np.asarray([len(x) for x in ids],dtype=np.int32)
    assert lengths.min() >= 2 and lengths.max() <= MAX_LENGTH
    return ToxicCommentDataset(ids,y), y, lengths, frame['id'].astype(str).to_numpy()

train_dataset,y_train,train_token_lengths,train_comment_ids = load_s02_dataset(S02_TRAIN_CACHE_PATH,FULL_TRAIN_PATH,TRAINING_SAMPLES)
val_dataset,y_val,val_token_lengths,val_comment_ids = load_s02_dataset(S02_VALIDATION_CACHE_PATH,VALIDATION_PATH,VALIDATION_SAMPLES)
assert len(set(train_comment_ids).intersection(val_comment_ids)) == 0
np.testing.assert_array_equal(y_train.sum(axis=0),[12232,1275,6757,382,6298,1121])
np.testing.assert_array_equal(y_val.sum(axis=0),[3059,319,1691,96,1578,284])
print('Full training and validation IDs and targets verified.')

INITIAL_MODEL_DIR = PROJECT_ROOT / s02_token_cfg['shared_initialization_path']
initialization_manifest = read_json(INITIAL_MODEL_DIR / 'initialization_manifest.json')
assert initialization_manifest['model_state_sha256'] == INITIAL_STATE_HASH
assert initialization_manifest['specification']['labels'] == LABELS
assert initialization_manifest['specification']['checkpoint'] == MODEL_CHECKPOINT
assert initialization_manifest['specification']['revision'] == MODEL_REVISION

def model_state_sha256(model):

    digest = hashlib.sha256()

    for name, tensor in sorted(
        model.state_dict().items()
    ):

        tensor = tensor.detach().cpu().contiguous()

        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('utf-8'))

        digest.update(
            tensor.numpy().tobytes()
        )

    return digest.hexdigest()


def verify_model_files(model_dir, manifest):

    assert manifest['files']

    for filename, expected_hash in manifest['files'].items():

        path = model_dir / filename

        assert path.is_file(), f"Missing model file: {path}"

        assert calculate_sha256(path) == expected_hash, (
            f"Model fingerprint mismatch: {path}"
        )


def persist_model_directory(
    model,
    destination,
    manifest_filename,
    metadata
):

    destination = Path(destination)

    assert not destination.exists(), (
        f"Model destination already exists: {destination}"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_destination = destination.with_name(
        destination.name + '.tmp'
    )

    assert not temporary_destination.exists(), (
        f"Incomplete model directory requires inspection: "
        f"{temporary_destination}"
    )

    with tempfile.TemporaryDirectory(
        dir='/content'
    ) as local_directory:

        local_path = Path(local_directory)

        model.save_pretrained(
            local_path,
            safe_serialization=True
        )

        tokenizer.save_pretrained(local_path)

        file_hashes = {
            str(path.relative_to(local_path)):
                calculate_sha256(path)
            for path in sorted(local_path.rglob('*'))
            if path.is_file()
        }

        assert any(
            name.endswith('.safetensors')
            for name in file_hashes
        )

        manifest = {
            **metadata,
            'model_state_sha256':
                model_state_sha256(model),
            'files': file_hashes
        }

        save_json_atomic(
            local_path / manifest_filename,
            manifest
        )

        shutil.copytree(
            local_path,
            temporary_destination
        )

        verify_model_files(
            temporary_destination,
            manifest
        )

        temporary_destination.rename(destination)

    return manifest


# Verify and copy the shared initialization locally; no download or fine-tuned initialization.
verify_model_files(INITIAL_MODEL_DIR, initialization_manifest)
LOCAL_INITIAL_MODEL_DIR = Path('/content/nlp04_s02_initialization_s42')
if not LOCAL_INITIAL_MODEL_DIR.exists():
    shutil.copytree(INITIAL_MODEL_DIR, LOCAL_INITIAL_MODEL_DIR)
verify_model_files(LOCAL_INITIAL_MODEL_DIR, initialization_manifest)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_INITIAL_MODEL_DIR, local_files_only=True, use_fast=True)
assert tokenizer.is_fast and tokenizer.pad_token_id == 1
assert tokenizer.padding_side == tokenizer.truncation_side == 'right'
assert tokenizer.vocab_size == 50265
print('Shared initialization files and tokenizer verified.')

def dynamic_collate_fn(items):
    features = [{'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']} for x in items]
    batch = tokenizer.pad(features, padding=True, pad_to_multiple_of=PAD_MULTIPLE, return_tensors='pt')
    batch['labels'] = torch.stack([x['labels'] for x in items])
    batch['row_index'] = torch.tensor([x['row_index'] for x in items],dtype=torch.long)
    assert batch['input_ids'].shape[1] <= MAX_LENGTH
    return batch

# Same numerically stable focal loss as T06, calculated in float32.
def compute_multilabel_loss(logits, targets, loss_name, reduction='mean'):
    assert loss_name == 'FOCAL'
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    loss = bce * (-torch.expm1(-bce)).pow(2.0)
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'none':
        return loss
    raise ValueError(reduction)

# Keep the reference signature and record full-data changes explicitly.
S02_RUN_SIGNATURE = copy.deepcopy(ref_sig)
S02_RUN_SIGNATURE.update({
    'run_id': S02_RUN_ID,
    'training_dataset_sha256': EXPECTED_FULL_TRAINING_HASH,
    'training_token_cache_sha256': s02_token_cfg['training_token_cache_sha256'],
    'warmup_steps': WARMUP_STEPS,
    'total_training_steps': TOTAL_TRAINING_STEPS,
    'gpu_name': torch.cuda.get_device_name(0),
    'torch_version': str(torch.__version__),
    'transformers_version': transformers.__version__,
    'numpy_version': np.__version__,
    'tokenizers_version': tokenizers.__version__,
    'determinism_policy': copy.deepcopy(policy),
    'determinism_policy_source': policy_source
})

def get_run_signature(run_id):
    assert run_id == S02_RUN_ID
    return copy.deepcopy(S02_RUN_SIGNATURE)


def load_fresh_model():

    set_random_seed(SEED)

    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_INITIAL_MODEL_DIR,
        local_files_only=True
    )

    assert model.config.num_labels == NUM_LABELS

    assert model.config.problem_type == (
        'multi_label_classification'
    )

    assert model.config.id2label == {
        index: label
        for index, label in enumerate(LABELS)
    }

    assert model_state_sha256(model) == INITIAL_STATE_HASH

    return model.to(DEVICE)


def create_training_components(
    total_steps=TOTAL_TRAINING_STEPS,
    warmup_steps=WARMUP_STEPS
):

    model = load_fresh_model()

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.endswith('.bias') or parameter.ndim == 1:

            no_decay_parameters.append(parameter)

        else:

            decay_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                'params': decay_parameters,
                'weight_decay': WEIGHT_DECAY
            },
            {
                'params': no_decay_parameters,
                'weight_decay': 0.0
            }
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=USE_FP16,
        init_scale=AMP_INITIAL_SCALE
    )

    return model, optimizer, scheduler, scaler


def get_epoch_indices(
    epoch,
    sample_count=TRAINING_SAMPLES
):

    rng = np.random.default_rng(
        SEED + epoch - 1
    )

    return rng.permutation(sample_count)


def create_training_loader(
    epoch,
    next_row=0,
    dataset=None
):

    if dataset is None:
        dataset = train_dataset

    indices = get_epoch_indices(
        epoch,
        len(dataset)
    )

    assert 0 <= next_row <= len(indices)

    remaining_indices = indices[next_row:].tolist()

    generator = torch.Generator()

    generator.manual_seed(
        SEED + epoch - 1
    )

    return DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dynamic_collate_fn,
        generator=generator
    )


def new_training_progress():

    return {
        'epoch': 1,
        'next_row': 0,
        'epoch_loss_sum': 0.0,
        'epoch_examples': 0,
        'attempted_updates': 0,
        'successful_updates': 0,
        'skipped_updates': 0,
        'best_macro_ap': None,
        'best_epoch': None,
        'best_checkpoint_path': None,
        'best_validation_path': None,
        'best_validation_sha256': None,
        'epochs_without_improvement': 0,
        'history': [],
        'training_time_seconds': 0.0,
        'training_finished': False,
        'stopping_reason': None
    }


def capture_rng_state():

    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all()
    }


def restore_rng_state(state):

    random.setstate(state['python'])

    np.random.set_state(state['numpy'])

    torch.set_rng_state(state['torch_cpu'])

    torch.cuda.set_rng_state_all(
        state['torch_cuda']
    )


def save_resume_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    progress,
    run_signature
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_drive_path = path.with_suffix(
        '.pt.tmp'
    )

    checkpoint = {
        'schema_version': 1,
        'run_signature': run_signature,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(),
        'progress': progress,
        'rng_state': capture_rng_state()
    }

    with tempfile.TemporaryDirectory(
        dir='/content'
    ) as local_directory:

        local_path = (
            Path(local_directory) / 'checkpoint.pt'
        )

        torch.save(
            checkpoint,
            local_path
        )

        shutil.copyfile(
            local_path,
            temporary_drive_path
        )

        assert temporary_drive_path.stat().st_size == (
            local_path.stat().st_size
        )

        assert calculate_sha256(temporary_drive_path) == (
            calculate_sha256(local_path)
        ), 'Checkpoint transfer fingerprint mismatch.'

        temporary_drive_path.replace(path)

    del checkpoint


def load_resume_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    run_signature
):

    path = Path(path)

    assert path.is_file()

    with tempfile.TemporaryDirectory(
        dir='/content'
    ) as local_directory:

        local_path = (
            Path(local_directory) / 'checkpoint.pt'
        )

        shutil.copyfile(path, local_path)

        assert calculate_sha256(local_path) == (
            calculate_sha256(path)
        ), "Checkpoint read-back fingerprint mismatch."

        checkpoint = torch.load(
            local_path,
            map_location='cpu',
            weights_only=False
        )

    assert checkpoint['schema_version'] == 1

    assert checkpoint['run_signature'] == run_signature, (
        "Checkpoint configuration differs from current configuration."
    )

    model.load_state_dict(
        checkpoint['model_state'],
        strict=True
    )

    optimizer.load_state_dict(
        checkpoint['optimizer_state']
    )

    scheduler.load_state_dict(
        checkpoint['scheduler_state']
    )

    scaler.load_state_dict(
        checkpoint['scaler_state']
    )

    progress = checkpoint['progress']

    restore_rng_state(
        checkpoint['rng_state']
    )

    del checkpoint

    gc.collect()

    return progress


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    scaler,
    loss_name,
    progress,
    checkpoint_callback=None,
    dataset=None,
    max_updates=None,
    verbose=True
):

    if dataset is None:
        dataset = train_dataset

    sample_count = len(dataset)

    model.train()

    epoch = progress['epoch']

    assert progress['next_row'] == (
        progress['epoch_examples']
    )

    assert (
        progress['next_row'] % EFFECTIVE_BATCH_SIZE == 0
        or progress['next_row'] == sample_count
    )

    loader = create_training_loader(
        epoch,
        progress['next_row'],
        dataset
    )

    loader_iterator = iter(loader)

    updates_this_call = 0

    while progress['next_row'] < sample_count:

        group_start_time = time.perf_counter()

        group_size = min(
            EFFECTIVE_BATCH_SIZE,
            sample_count - progress['next_row']
        )

        microbatches = math.ceil(
            group_size / TRAIN_BATCH_SIZE
        )

        optimizer.zero_grad(set_to_none=True)

        group_loss_sum = 0.0
        processed_samples = 0

        for _ in range(microbatches):

            batch = next(loader_iterator)

            input_ids = batch['input_ids'].to(
                DEVICE,
                non_blocking=True
            )

            attention_mask = batch['attention_mask'].to(
                DEVICE,
                non_blocking=True
            )

            targets = batch['labels'].to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=USE_FP16
            ):

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits

            loss_sum = compute_multilabel_loss(
                logits,
                targets,
                loss_name,
                reduction='sum'
            )

            assert torch.isfinite(loss_sum), (
                "Non-finite training loss encountered."
            )

            # Normalize using actual samples in the accumulation group

            normalized_loss = (
                loss_sum / (group_size * NUM_LABELS)
            )

            scaler.scale(
                normalized_loss
            ).backward()

            group_loss_sum += loss_sum.detach().item()

            processed_samples += targets.shape[0]

        assert processed_samples == group_size

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )

        previous_scale = scaler.get_scale()

        scaler.step(optimizer)
        scaler.update()

        current_scale = scaler.get_scale()

        progress['attempted_updates'] += 1

        if current_scale >= previous_scale:

            scheduler.step()

            progress['successful_updates'] += 1

        else:

            progress['skipped_updates'] += 1

        optimizer.zero_grad(set_to_none=True)

        progress['next_row'] += processed_samples

        progress['epoch_examples'] += processed_samples

        progress['epoch_loss_sum'] += group_loss_sum

        torch.cuda.synchronize()

        progress['training_time_seconds'] += (
            time.perf_counter() - group_start_time
        )

        updates_this_call += 1

        update_number = progress['attempted_updates']

        if verbose and (
            update_number % LOG_INTERVAL == 0
        ):

            running_loss = (
                progress['epoch_loss_sum']
                / (progress['epoch_examples'] * NUM_LABELS)
            )

            print(
                f"Epoch {epoch} | "
                f"Rows {progress['next_row']:,}/{sample_count:,} | "
                f"Update {update_number:,} | "
                f"Loss {running_loss:.6f}",
                flush=True
            )

        if (
            checkpoint_callback is not None
            and update_number % CHECKPOINT_INTERVAL == 0
        ):

            checkpoint_callback(progress)

        if (
            max_updates is not None
            and updates_this_call >= max_updates
            and progress['next_row'] < sample_count
        ):

            return None

    assert progress['epoch_examples'] == sample_count

    assert (
        progress['successful_updates']
        + progress['skipped_updates']
        == progress['attempted_updates']
    )

    return (
        progress['epoch_loss_sum']
        / (sample_count * NUM_LABELS)
    )


@torch.inference_mode()
def evaluate_validation_model(
    model,
    loss_name,
    dataset=None,
    targets=None,
    compute_metrics=True
):

    if dataset is None:
        dataset = val_dataset

    if targets is None:
        targets = y_val

    sample_count = len(dataset)

    targets = np.asarray(targets)

    assert targets.shape == (
        sample_count,
        NUM_LABELS
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dynamic_collate_fn,
        generator=generator
    )

    model.eval()

    all_logits = np.empty(
        (sample_count, NUM_LABELS),
        dtype=np.float32
    )

    all_scores = np.empty_like(all_logits)

    seen = np.zeros(
        sample_count,
        dtype=bool
    )

    validation_loss_sum = 0.0

    start_time = time.perf_counter()

    for batch in loader:

        row_indices = batch['row_index'].numpy()

        assert len(np.unique(row_indices)) == len(row_indices)

        assert not seen[row_indices].any()

        batch_targets = batch['labels']

        np.testing.assert_array_equal(
            batch_targets.numpy(),
            targets[row_indices]
        )

        input_ids = batch['input_ids'].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch['attention_mask'].to(
            DEVICE,
            non_blocking=True
        )

        batch_targets = batch_targets.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=USE_FP16
        ):

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

        logits = logits.float()

        scores = torch.sigmoid(logits)

        loss_sum = compute_multilabel_loss(
            logits,
            batch_targets,
            loss_name,
            reduction='sum'
        )

        assert torch.isfinite(logits).all()
        assert torch.isfinite(scores).all()
        assert torch.isfinite(loss_sum)

        all_logits[row_indices] = (
            logits.cpu().numpy()
        )

        all_scores[row_indices] = (
            scores.cpu().numpy()
        )

        validation_loss_sum += loss_sum.item()

        seen[row_indices] = True

    assert seen.all()

    assert np.isfinite(all_logits).all()
    assert np.isfinite(all_scores).all()

    assert (
        (all_scores >= 0)
        & (all_scores <= 1)
    ).all()

    metrics = None
    per_label = None

    if compute_metrics:

        metrics, per_label = evaluate_multilabel(
            targets,
            all_scores,
            LABELS,
            thresholds=0.5
        )

    torch.cuda.synchronize()

    return {
        'logits': all_logits,
        'scores': all_scores,
        'metrics': metrics,
        'per_label_metrics': per_label,
        'validation_loss':
            validation_loss_sum / (sample_count * NUM_LABELS),
        'inference_time_seconds':
            time.perf_counter() - start_time
    }


def save_best_model(
    model,
    run_model_dir,
    epoch
):

    timestamp = datetime.now(timezone.utc).strftime(
        '%Y%m%d_%H%M%S_%f'
    )

    destination = (
        Path(run_model_dir)
        / f'best_epoch_{epoch:02d}_{timestamp}'
    )

    persist_model_directory(
        model,
        destination,
        'checkpoint_manifest.json',
        {
            'epoch': epoch,
            'labels': LABELS,
            'seed': SEED,
            'initial_model_sha256': INITIAL_STATE_HASH,
            'created_at':
                datetime.now(timezone.utc).isoformat()
        }
    )

    return destination


def optimizer_state_sha256(optimizer):

    state = optimizer.state_dict()

    digest = hashlib.sha256()

    digest.update(
        json.dumps(
            state['param_groups'],
            sort_keys=True
        ).encode('utf-8')
    )

    for parameter_id, values in sorted(
        state['state'].items()
    ):

        digest.update(
            str(parameter_id).encode('utf-8')
        )

        for name, value in sorted(values.items()):

            digest.update(name.encode('utf-8'))

            if torch.is_tensor(value):

                tensor = (
                    value.detach()
                    .cpu()
                    .contiguous()
                )

                digest.update(
                    str(tensor.dtype).encode('utf-8')
                )

                digest.update(
                    str(tuple(tensor.shape)).encode('utf-8')
                )

                digest.update(
                    tensor.numpy().tobytes()
                )

            else:

                digest.update(
                    repr(value).encode('utf-8')
                )

    return digest.hexdigest()


def assert_rng_states_equal(expected, actual):
    assert expected['python'] == actual['python'], 'Python RNG mismatch'
    a, b = expected['numpy'], actual['numpy']
    assert a[0] == b[0] and a[2:] == b[2:], 'NumPy RNG metadata mismatch'
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(expected['torch_cpu'], actual['torch_cpu']), 'CPU RNG mismatch'
    assert len(expected['torch_cuda']) == len(actual['torch_cuda'])
    for x, y in zip(expected['torch_cuda'], actual['torch_cuda']):
        assert torch.equal(x, y), 'CUDA RNG mismatch'


# Exercise worst-case padding, the final 12-row accumulation group and a checkpoint round trip.
def s02_smoke_test():
    rng = np.random.default_rng(SEED)
    selected = rng.choice(np.flatnonzero(train_token_lengths == MAX_LENGTH),32,replace=False)
    other = rng.choice(np.setdiff1d(np.arange(TRAINING_SAMPLES),selected),12,replace=False)
    order = get_epoch_indices(1,44)
    arranged = np.empty(44,dtype=np.int64)
    arranged[order] = np.concatenate([selected,other])
    smoke_train = ToxicCommentDataset([train_dataset.input_ids[i] for i in arranged],y_train[arranged])
    val_indices = rng.choice(np.flatnonzero(val_token_lengths == MAX_LENGTH),32,replace=False)
    smoke_val = ToxicCommentDataset([val_dataset.input_ids[i] for i in val_indices],y_val[val_indices])
    batch = next(iter(create_training_loader(1,dataset=smoke_train)))
    assert tuple(batch['input_ids'].shape) == (16,256)
    assert batch['attention_mask'].eq(1).all()
    vb = dynamic_collate_fn([smoke_val[i] for i in range(32)])
    assert tuple(vb['input_ids'].shape) == (32,256)
    assert vb['attention_mask'].eq(1).all()
    print('Maximum-length train and validation batches verified.')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    smoke_sig = {**get_run_signature(S02_RUN_ID),'purpose':'s02_smoke_only','smoke_training_samples':44,'smoke_total_steps':2,'smoke_warmup_steps':1}
    model,optimizer,scheduler,scaler = create_training_components(total_steps=2,warmup_steps=1)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 124650246
    assert model_state_sha256(model) == INITIAL_STATE_HASH
    progress = new_training_progress()
    result = train_one_epoch(model,optimizer,scheduler,scaler,'FOCAL',progress,dataset=smoke_train,max_updates=1,verbose=False)
    assert result is None and progress['next_row'] == 32
    assert progress['successful_updates'] == 1 and progress['skipped_updates'] == 0
    with tempfile.TemporaryDirectory(dir='/content') as td:
        path = Path(td)/'smoke_checkpoint.pt'
        save_resume_checkpoint(path,model,optimizer,scheduler,scaler,progress,smoke_sig)
        expected_hash = calculate_sha256(path)
        expected_model = model_state_sha256(model)
        expected_optimizer = optimizer_state_sha256(optimizer)
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        expected_scaler = copy.deepcopy(scaler.state_dict())
        expected_progress = copy.deepcopy(progress)
        expected_rng = capture_rng_state()
        print('Smoke checkpoint saved at row 32.')
        # Use the shared prefix for the continuation comparison.
        ref_loss = train_one_epoch(model,optimizer,scheduler,scaler,'FOCAL',progress,dataset=smoke_train,verbose=False)
        ref_val = evaluate_validation_model(model,'FOCAL',dataset=smoke_val,targets=smoke_val.targets,compute_metrics=False)
        ref_progress = copy.deepcopy(progress)
        del model,optimizer,scheduler,scaler
        gc.collect(); torch.cuda.empty_cache()
        model,optimizer,scheduler,scaler = create_training_components(total_steps=2,warmup_steps=1)
        assert calculate_sha256(path) == expected_hash
        progress = load_resume_checkpoint(path,model,optimizer,scheduler,scaler,smoke_sig)
        assert model_state_sha256(model) == expected_model
        assert optimizer_state_sha256(optimizer) == expected_optimizer
        assert scheduler.state_dict() == expected_scheduler and scaler.state_dict() == expected_scaler
        assert progress == expected_progress
        assert_rng_states_equal(expected_rng,capture_rng_state())
        print('Model, optimizer, scheduler, scaler, progress and RNG restored exactly.')
        resumed_loss = train_one_epoch(model,optimizer,scheduler,scaler,'FOCAL',progress,dataset=smoke_train,verbose=False)
        resumed_val = evaluate_validation_model(model,'FOCAL',dataset=smoke_val,targets=smoke_val.targets,compute_metrics=False)
        assert progress['next_row'] == 44 and progress['successful_updates'] == 2 and progress['skipped_updates'] == 0
        # Timing and accumulated floating-point loss are not exact-equality gates.
        for key in progress:
            if key not in ('training_time_seconds', 'epoch_loss_sum'):
                assert progress[key] == ref_progress[key], key
        loss_diff = abs(ref_loss-resumed_loss)
        logit_diff = float(np.max(np.abs(ref_val['logits']-resumed_val['logits'])))
        score_diff = float(np.max(np.abs(ref_val['scores']-resumed_val['scores'])))
        assert loss_diff <= 1e-5 and logit_diff <= 2e-3 and score_diff <= 5e-4, 'Numerical continuation check failed.'
        result = {'shared_initialization_verified': True,'exact_checkpoint_restore_verified':True,
                  'numerical_continuation_verified':True,'bitwise_continuation_claimed':False,
                  'training_smoke_samples':44,'validation_smoke_samples':32,'final_smoke_group_size':12,
                  'smoke_successful_updates':2,'smoke_skipped_updates':0,
                  'continuation_loss_difference':float(loss_diff),'maximum_logit_difference':logit_diff,
                  'maximum_probability_difference':score_diff,
                  'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                  'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                  'checkpoint_size_mib':path.stat().st_size/1024**2,
                  'temporary_smoke_checkpoint_retained':False,'official_test_used':False}
        del model,optimizer,scheduler,scaler
    gc.collect(); torch.cuda.empty_cache()
    return result

S02_PIPELINE_CONFIG_PATH = RUN_CONFIGS_DIR / 's02_training_pipeline.json'
S02_SMOKE_SUMMARY_PATH = LOGS_DIR / 's02_smoke_test_summary.json'
s02_smoke_summary = s02_smoke_test()
save_json_atomic(S02_SMOKE_SUMMARY_PATH,s02_smoke_summary)
s02_pipeline_configuration = {
    'run_id':S02_RUN_ID,'reference_run_id':'T06_FOCAL_S42','run_signature':get_run_signature(S02_RUN_ID),
    'reference_configuration_sha256':calculate_sha256(reference_config_path),
    'reference_pipeline_sha256':calculate_sha256(previous_pipeline_path),
    'training_samples':TRAINING_SAMPLES,'validation_samples':VALIDATION_SAMPLES,
    'batches_per_epoch':BATCHES_PER_EPOCH,'updates_per_epoch':UPDATES_PER_EPOCH,
    'total_training_steps':TOTAL_TRAINING_STEPS,'warmup_steps':WARMUP_STEPS,
    'final_accumulation_group_size':FINAL_ACCUMULATION_SIZE,
    'initial_model_state_sha256':INITIAL_STATE_HASH,'local_checkpoint_interval_updates':CHECKPOINT_INTERVAL,
    'drive_checkpoint_policy':'epoch boundary; keep latest recovery state only',
    'smoke_summary_path':str(S02_SMOKE_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
    'smoke_summary_sha256':calculate_sha256(S02_SMOKE_SUMMARY_PATH),
    'training_configuration_finalized':True,'checkpoint_resume_verified':True,
    'full_experiment_training_performed':False,'official_test_used':False,'final_project_model_frozen':False
}
save_json_atomic(S02_PIPELINE_CONFIG_PATH,s02_pipeline_configuration)
assert calculate_sha256(REGISTRY_PATH)==registry_hash_before
assert calculate_sha256(DECISION_LOG_PATH)==decision_hash_before
print('\nS02 training pipeline summary')
for k,v in s02_smoke_summary.items(): print(k,':',v)
print('Full training rows:',TRAINING_SAMPLES)
print('Updates per epoch:',UPDATES_PER_EPOCH)
print('Maximum updates:',TOTAL_TRAINING_STEPS)
print('Warmup updates:',WARMUP_STEPS)
print('Final accumulation group:',FINAL_ACCUMULATION_SIZE)
print('Shared initial model SHA256:',INITIAL_STATE_HASH)
print('Full training performed: No')
print('Experiment registry: Unchanged')
print('Decision log: Unchanged')
print('Official test dataset: Not loaded')
print('Cell 6 completed successfully.')
