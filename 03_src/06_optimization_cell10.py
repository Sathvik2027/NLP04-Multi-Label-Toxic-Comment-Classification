# Notebook 06, Cell 10: freeze the validation-selected evaluation roster
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

freeze_path = RUN_CONFIGS_DIR / 'frozen_manifest.json'
analysis_path = RUN_CONFIGS_DIR / 'final_validation_analysis.json'
a = read_json(analysis_path)
reg = pd.read_csv(REGISTRY_PATH, dtype=str, keep_default_na=False)
assert len(reg) == a['completed_runs'] == 19
assert reg['run_id'].is_unique and reg['status'].eq('Completed').all()
assert not a['official_test_used'] and not a['thresholds_changed']
assert calculate_sha256(REGISTRY_PATH) == a['registry_sha256']
assert calculate_sha256(VALIDATION_PATH) == a['validation_dataset_sha256']
assert calculate_sha256(SRC_DIR/'evaluation_utils.py') == a['evaluation_module_sha256']
for name, expected in a['artifacts'].items():
    assert calculate_sha256(TABLES_DIR/name) == expected, name
ranking = pd.read_csv(TABLES_DIR/'T01_model_selection.csv', float_precision='round_trip')
threshold_table = pd.read_csv(TABLES_DIR/'T04_selected_thresholds.csv', float_precision='round_trip')
assert set(ranking['Run ID']) == set(reg['run_id'])
assert len(ranking) == 19 and len(threshold_table) == 19*len(LABELS)
assert ranking['Macro AP'].is_monotonic_decreasing
assert ranking.iloc[0]['Run ID'] == a['selected_run_id'] == 'S02_ROBERTA_FOCAL_FULL_S42'
assert len(a['sources']) == 19
source_by_run = {s['run_id']:s for s in a['sources']}
assert set(source_by_run) == set(reg['run_id'])

# Resolve only files inside the active project and avoid duplicate hashing.
root10 = PROJECT_ROOT.resolve()
_hashes10 = {}
def rel10(path):
    return str(Path(path).resolve().relative_to(root10))
def file10(path, expected=None):
    path = Path(path).resolve()
    relative = rel10(path)
    assert path.is_file() and path.stat().st_size > 0, path
    if relative not in _hashes10:
        _hashes10[relative] = calculate_sha256(path)
    digest = _hashes10[relative]
    if expected is not None:
        assert digest == expected, f'Fingerprint mismatch: {path}'
    return {'path':relative,'sha256':digest,'size_bytes':path.stat().st_size}

def recorded10(value, *parents):
    assert isinstance(value,str) and value
    paths = {p.resolve() for p in [PROJECT_ROOT/value, *[parent/value for parent in parents]] if p.exists()}
    assert len(paths)==1, f'Cannot resolve {value}: {paths}'
    p = paths.pop(); rel10(p)
    return p

# Match prior feature manifests by their recorded hash, not file naming alone.
feature_manifests10 = {}
def feature_manifest10(expected):
    if expected not in feature_manifests10:
        candidates = list(FEATURES_DIR.rglob('*manifest*.csv')) + list(FEATURES_DIR.rglob('*manifest*.json'))
        matches = [p for p in candidates if file10(p)['sha256']==expected]
        assert len(matches)==1, f'Cannot locate classical feature manifest {expected}: {matches}'
        feature_manifests10[expected] = matches[0]
    return feature_manifests10[expected]

# Find evidence that a particular saved model or vectorizer has its original hash.
def prior_file_evidence10(path, candidates, rid=None):
    rec = file10(path)
    expected_path = rec['path']
    for p in candidates:
        if not p.is_file(): continue
        if p.suffix.lower()=='.csv':
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            for row in df.to_dict('records'):
                vals = [str(v).replace('\\','/') for v in row.values()]
                if rid and row.get('Run ID',row.get('run_id',rid)) != rid: continue
                # Require the exact relative path or a basename match with the exact digest.
                if expected_path in vals:
                    hashvals=[v for v in vals if len(v)==64 and all(c in '0123456789abcdef' for c in v.lower())]
                    if hashvals:
                        assert rec['sha256'] in hashvals, f'Historical hash mismatch: {path}'
                        return file10(p)
                if rec['sha256'] in vals and any(v==path.name or v.endswith('/'+path.name) for v in vals):
                    return file10(p)
        elif p.suffix.lower()=='.json':
            obj=read_json(p)
            # Preserve the original nested structure when locating the paired path and hash.
            def visit(value):
                if isinstance(value,dict):
                    vals=[v for v in value.values() if isinstance(v,str)]
                    if expected_path in vals and any(len(v)==64 and all(c in '0123456789abcdef' for c in v.lower()) for v in vals):
                        # Only compare hashes on records with a file-specific hash key.
                        for k in ('sha256','SHA256','model_sha256','file_sha256'):
                            if k in value:
                                assert value[k]==rec['sha256'], f'Historical hash mismatch: {path}'
                                return True
                    if rec['sha256'] in vals and any(v==expected_path or v==path.name or v.endswith('/'+path.name) for v in vals): return True
                    return any(visit(v) for v in value.values())
                if isinstance(value,list): return any(visit(v) for v in value)
                return False
            if visit(obj): return file10(p)
    return None

# Freeze the exact model dependencies without copying model weights.
entries10=[]; issues10=[]
for row in reg.to_dict('records'):
    rid=row['run_id']; src=source_by_run[rid]
    try:
        cfg_path=PROJECT_ROOT/row['config_path']; cfg=read_json(cfg_path)
        assert cfg['run_id']==rid and cfg['official_test_used'] is False
        cfg_rec=file10(cfg_path)
        model_path=PROJECT_ROOT/row['model_path']
        result_root=PROJECT_ROOT/row['results_path']
        pred=file10(PROJECT_ROOT/src['predictions_path'],src['predictions_sha256'])
        thresh=file10(PROJECT_ROOT/src['threshold_path'],src['threshold_sha256'])
        tc=read_json(PROJECT_ROOT/thresh['path'])
        t=threshold_table.loc[threshold_table['Run ID'].eq(rid)].set_index('Label').loc[LABELS]
        raw=tc.get('optimized_thresholds',tc.get('thresholds'))
        assert isinstance(raw,dict) and set(raw)==set(LABELS)
        cutoffs={l:float(raw[l]) for l in LABELS}
        np.testing.assert_allclose(list(cutoffs.values()),t['Optimized Threshold'].to_numpy(),rtol=0,atol=1e-12)
        assert t['Score Type'].nunique()==1 and t['Default Threshold'].nunique()==1
        default=float(t['Default Threshold'].iloc[0]); score_type=str(t['Score Type'].iloc[0])
        metric=ranking.loc[ranking['Run ID'].eq(rid)].iloc[0]
        assert abs(float(row['validation_macro_ap'])-float(metric['Macro AP']))<1e-6
        assert abs(float(row['validation_macro_f1'])-float(metric['Optimized Macro F1']))<1e-6
        is_transformer=model_path.is_dir()
        entry={'run_id':rid,'model':row['model'],'feature_representation':row['feature_representation'],
               'sampling_strategy':row['sampling_strategy'],'training_samples':int(float(row['training_samples'])),
               'seed':int(float(row['seed'])),'model_path':rel10(model_path),'config':cfg_rec,
               'validation_predictions':pred,'threshold_file':thresh,'labels':list(LABELS),
               'score_type':score_type,'default_threshold':default,'optimized_thresholds':cutoffs,
               'comparison_operator':'>=','validation_macro_ap':float(metric['Macro AP']),
               'validation_optimized_macro_f1':float(metric['Optimized Macro F1']),
               'training_time_seconds':float(metric['Training Time Seconds']),
               'kind':'transformer' if is_transformer else 'classical'}
        if is_transformer:
            sig=cfg['run_signature']; assert sig['labels']==LABELS
            cm_path=model_path/'checkpoint_manifest.json'; cm=read_json(cm_path)
            assert cm['files'] and cm['labels']==LABELS
            files=[file10(model_path/name,digest) for name,digest in cm['files'].items()]
            assert any(x['path'].endswith('.safetensors') for x in files)
            assert any(x['path'].endswith('/tokenizer.json') for x in files)
            entry.update({'model_files':files,'checkpoint_manifest':file10(cm_path),
                'model_state_sha256':cm['model_state_sha256'],'checkpoint':sig['checkpoint'],
                'revision':sig['revision'],'maximum_length':int(sig['maximum_length']),
                'padding_multiple':int(sig['padding_multiple']),'inference_precision':'fp16_cuda',
                'training_dataset_sha256':sig['training_dataset_sha256'],
                'validation_dataset_sha256':sig['validation_dataset_sha256']})
            assert sig['validation_dataset_sha256']==a['validation_dataset_sha256']
        else:
            mrec=file10(model_path,cfg.get('model_sha256'))
            candidates=list(result_root.rglob('*manifest*.csv'))+list(result_root.rglob('*verification*.csv'))+list(result_root.rglob('*summary*.json'))+[cfg_path]
            # The final selection records provide independent fingerprints for the two selected classical models.
            if CLASSICAL_SELECTION_PATH.is_file(): candidates.append(CLASSICAL_SELECTION_PATH)
            prior=prior_file_evidence10(model_path,candidates,rid)
            entry['model_files']=[mrec]
            entry['prior_model_hash_evidence']=prior
            # A missing historical hash is explicitly distinguished from a newly frozen fingerprint.
            entry['model_hash_previously_verified']=bool(prior or cfg.get('model_sha256'))
            entry['training_dataset_sha256']=cfg['training_dataset_sha256']
            entry['validation_dataset_sha256']=cfg['validation_dataset_sha256']
            assert cfg['validation_dataset_sha256']==a['validation_dataset_sha256']
            if rid.startswith('S01_'):
                fm_path=recorded10(cfg['feature_manifest_path'],result_root,model_path.parent)
                fm=read_json(fm_path)
                entry['feature_manifest']=file10(fm_path,cfg['feature_manifest_sha256'])
                deps=[]
                for key in ('word','character'):
                    dep=fm['vectorizers'][key]
                    vpath=recorded10(dep['path'],fm_path.parent)
                    deps.append({'role':key,**file10(vpath,dep['sha256'])})
                entry['feature_mode']='combined'
            else:
                fm_path=feature_manifest10(cfg['feature_manifest_sha256'])
                entry['feature_manifest']=file10(fm_path,cfg['feature_manifest_sha256'])
                mode='combined' if '_COMBINED_' in rid else 'character' if '_CHAR_' in rid else 'word' if '_WORD_' in rid else None
                assert mode is not None,rid
                vdir=PROJECT_ROOT/'04_models/classical/vectorizers'
                specs={'combined':[('word','combined_word_tfidf.joblib'),('character','combined_character_tfidf.joblib')],
                       'character':[('character','character_tfidf.joblib')],'word':[('word','word_tfidf.joblib')]}[mode]
                deps=[]
                for role,name in specs:
                    vpath=vdir/name
                    prior_v=prior_file_evidence10(vpath,[fm_path,CLASSICAL_SELECTION_PATH])
                    assert prior_v is not None, f'No recorded vectorizer hash for {vpath}'
                    deps.append({'role':role,**file10(vpath)})
                entry['feature_mode']=mode
            entry['vectorizers']=deps
            entry['combined_feature_order']=['word','character'] if entry['feature_mode']=='combined' else None
            entry['combined_final_l2_normalization']=entry['feature_mode']=='combined'
            entry['inference_method']='decision_function' if rid.startswith('SVM_') else 'predict_proba'
        entries10.append(entry)
        print('Freeze dependencies verified:',rid,flush=True)
    except Exception as exc:
        issues10.append(f'{rid}: {type(exc).__name__}: {exc}')
if issues10:
    print('\nFreeze preflight issues:')
    for issue in issues10: print(issue)
    raise RuntimeError('Resolve preflight issues before freezing. No freeze or decision updates written.')
assert len(entries10)==19

# Read only the prepared dataset manifest, not the official test dataset.
dm=pd.read_csv(SPLITS_DIR/'prepared_dataset_manifest.csv',dtype=str,keep_default_na=False)
test_rows=dm.loc[dm['Partition'].eq('test_official')]
if test_rows.empty:
    test_rows=dm.loc[dm['Filename'].eq('test_official.parquet')]
assert len(test_rows)==1, 'Official test metadata must be unique'
test_record=test_rows.iloc[0]
assert int(test_record['Rows'])==63978
assert len(test_record['SHA256'])==64

# Document the resource-driven scope revision before unlocking test evaluation.
selected={'primary_model':a['selected_run_id'],'representative_transformer':'T06_FOCAL_S42',
          'lightweight_transformer':'T03_FOCAL_S42','best_classical':'S01_LR_COMBINED_BALANCED_FULL_S42',
          'representative_classical':'LR_COMBINED_BALANCED_S42','secondary_classical':'SVM_COMBINED_BASELINE_S42'}
assert set(selected.values()).issubset(source_by_run)
figure_paths=[ROBERTA_RESULTS_DIR/'figures/transformer_consolidated_comparison.png',
              FIGURES_DIR/'F02_scale_comparison.png',FIGURES_DIR/'F03_F1_threshold_panels.png',
              FIGURES_DIR/'F04_default_vs_tuned_validation.png']
figure_records=[file10(p) for p in figure_paths]
changes=[('D069','Full-data scaling','Completed S01 and S02; retain validation-selected best epochs and thresholds.',
          'Both high-priority full-data runs completed; scaling changes both data volume and optimizer updates.','Completed'),
         ('D070','Execution policy','Record unconfirmed historical S02 determinism policy.',
          'S02 explicit deterministic execution was verified by its smoke test; exact historical policy was unavailable.','Recorded'),
         ('D071','Threshold stability','Retain original six validation-optimized S02 thresholds.',
          '200 group bootstrap replicates per label; threat threshold widest. No bootstrap-based threshold change.','Completed'),
         ('D072','Resource allocation','Defer S03, R01, A01, A02 and A03.',
          'GPU quota, elapsed training time and deadline; mandatory test, fairness, explanation and delivery take priority.','Deferred'),
         ('D073','Model selection','Select S02 by validation Macro AP and freeze all 19 completed runs.',
          'Validation-only ranking reconstructed from saved predictions; test results not used for selection.','Completed'),
         ('D074','Evaluation boundary','No post-freeze tuning or model replacement based on official test.',
          'Later extensions must be separate exploratory runs; historical team test exposure disclosed.','Recorded')]
new_decisions=pd.DataFrame(changes,columns=['decision_id','component','decision','justification','status'])
decisions10=pd.read_csv(DECISION_LOG_PATH,dtype=str,keep_default_na=False)
assert decisions10['decision_id'].is_unique
assert set(new_decisions.columns)==set(decisions10.columns), list(decisions10.columns)
old_ids=set(decisions10['decision_id']); new_ids=set(new_decisions['decision_id'])
if new_ids & old_ids:
    assert new_ids.issubset(old_ids), 'Partial decision update; inspect before proceeding'
    for record in new_decisions.to_dict('records'):
        saved=decisions10.loc[decisions10['decision_id'].eq(record['decision_id'])].iloc[0].to_dict()
        assert saved==record,record['decision_id']
    final_decisions=decisions10
else:
    assert len(decisions10)==68
    final_decisions=pd.concat([decisions10,new_decisions[decisions10.columns]],ignore_index=True)
assert len(final_decisions)==74
assert calculate_sha256(REGISTRY_PATH)==a['registry_sha256']

# The final manifest is a lock file; verify instead of overwriting on reruns.
manifest10={'schema_version':1,'project':'NLP04_Toxic_Comment_Classification','labels':list(LABELS),
    'selection_partition':'validation','primary_selection_metric':'macro_average_precision','selected_models':selected,
    'primary_selected_run_id':a['selected_run_id'],'evaluation_roster':ranking['Run ID'].tolist(),
    'completed_experiments':19,'models':entries10,'registry':file10(REGISTRY_PATH),
    'validation_analysis':file10(analysis_path),'validation_dataset_sha256':a['validation_dataset_sha256'],
    'evaluation_module':file10(SRC_DIR/'evaluation_utils.py'),
    'required_validation_tables':[file10(TABLES_DIR/name) for name in a['artifacts']],
    'validation_figures':figure_records,'official_test':{'dataset_path':test_record['Dataset Path'],
        'expected_rows':63978,'expected_sha256':test_record['SHA256'],'evaluated':False},
    'deferred_experiments':a['deferred_experiments'],'defer_reason':a['defer_reason'],
    'limitations':['Single training seed; comparisons exploratory with respect to training variability.',
      'Validation used for checkpoint selection and threshold tuning; development metrics are not independent test estimates.',
      'Full-data scaling changes both data and optimization budget.',
      'Historical S02 execution policy unavailable for exact matching.',
      'Official benchmark previously examined by the team; held out for this pipeline only.',
      'Scores not established as calibrated probabilities.'],
    'thresholds_frozen':True,'model_parameters_frozen':True,'final_project_model_frozen':True,
    'official_test_used_for_selection':False,'test_evaluation_permitted':True,'post_freeze_refitting_permitted':False}
if freeze_path.exists():
    saved=read_json(freeze_path)
    assert saved.get('decision_log_sha256')==calculate_sha256(DECISION_LOG_PATH)
    assert {k:v for k,v in saved.items() if k not in ('frozen_at','decision_log_sha256')}==manifest10, 'Existing freeze differs; do not overwrite.'
    manifest10=saved
    print('Existing frozen manifest verified.')
else:
    if not new_ids.issubset(old_ids):
        save_csv_atomic(DECISION_LOG_PATH,final_decisions)
    pd.testing.assert_frame_equal(pd.read_csv(DECISION_LOG_PATH,dtype=str,keep_default_na=False),final_decisions,check_exact=True)
    manifest10['decision_log_sha256']=calculate_sha256(DECISION_LOG_PATH)
    manifest10['frozen_at']=datetime.now(timezone.utc).isoformat()
    save_json_atomic(freeze_path,manifest10)
    assert read_json(freeze_path)==manifest10

summary10={'status':'Completed','frozen_manifest_path':rel10(freeze_path),'frozen_manifest_sha256':calculate_sha256(freeze_path),
    'primary_selected_run_id':a['selected_run_id'],'completed_experiments':19,'methodological_decisions':74,
    'test_evaluation_roster_size':19,'deferred_experiments':a['deferred_experiments'],
    'classical_models_with_new_fingerprint_only':[e['run_id'] for e in entries10 if e['kind']=='classical' and not e['model_hash_previously_verified']],
    'additional_training_performed':False,'official_test_dataset_loaded':False,'final_project_model_frozen':True}
completion10=LOGS_DIR/'optimization_completion_summary.json'
if completion10.exists(): assert read_json(completion10)==summary10
else: save_json_atomic(completion10,summary10)
print('\nNotebook 06 completion summary')
for k,v in summary10.items(): print(f'{k}: {v}')
print('Frozen cutoffs:',len(entries10)*len(LABELS))
print('Official scored test population: 63978 (manifest only)')
print('Next notebook: 07_final_evaluation')
print('Cell 10 completed successfully.')
