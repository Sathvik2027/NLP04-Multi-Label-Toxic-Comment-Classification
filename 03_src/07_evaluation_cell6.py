# Notebook 07, Cell 6: test comparison figures from saved predictions
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

assert 'evaluation_setup' in globals(), 'Run Notebook 07 Cell 1 first.'
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(TEST_PATH) == EXPECTED_TEST_HASH
metrics_path6 = LOGS_DIR / 'test_metrics_summary.json'
metrics_hash6 = calculate_sha256(metrics_path6)
summary6 = read_json(metrics_path6)
assert summary6['status'] == 'Completed'
assert summary6['frozen_manifest_sha256'] == EXPECTED_FROZEN_HASH
assert summary6['test_dataset_sha256'] == EXPECTED_TEST_HASH
assert summary6['evaluated_runs'] == list(EVALUATION_ROSTER)
assert summary6['primary_selected_run_id'] == PRIMARY_RUN_ID
assert not summary6['thresholds_changed'] and not summary6['test_driven_selection_performed']
assert summary6['aggregate_records'] == 38 and summary6['per_label_records'] == 228
for rec in summary6['artifacts'] + summary6['inference_sources']:
    p = PROJECT_ROOT / rec['path']
    assert calculate_sha256(p) == rec['sha256'], str(p)

comparison6 = pd.read_csv(TABLES_DIR / 'test_model_comparison.csv').sort_values('Validation Rank').reset_index(drop=True)
labels6 = pd.read_csv(TABLES_DIR / 'test_per_label_metrics.csv')
index6 = pd.read_csv(TABLES_DIR / 'test_prediction_index.csv')
assert comparison6['Run ID'].tolist() == list(EVALUATION_ROSTER)
assert len(labels6) == 228 and len(index6) == 19
assert index6['Run ID'].is_unique and set(index6['Run ID']) == set(EVALUATION_ROSTER)
assert not labels6.duplicated(['Run ID','Operating Point','Label']).any()
primary6 = labels6.loc[(labels6['Run ID'] == PRIMARY_RUN_ID) & (labels6['Operating Point'] == 'optimized')].set_index('Label').loc[LABELS]
assert len(primary6) == 6
np.testing.assert_allclose(primary6['Threshold'], [FROZEN_MODELS[PRIMARY_RUN_ID]['optimized_thresholds'][l] for l in LABELS], rtol=0, atol=1e-12)
np.testing.assert_array_equal(primary6['Positive Support'], y_test.sum(axis=0))
assert abs(primary6['F1'].mean() - summary6['primary_test_optimized_macro_f1']) < 1e-12

# Verify primary scores and ID alignment before plotting PR curves.
r6 = index6.set_index('Run ID').loc[PRIMARY_RUN_ID]
pp6 = PROJECT_ROOT / r6['Prediction Path']
assert calculate_sha256(pp6) == r6['SHA256']
pred6 = pd.read_parquet(pp6)
assert len(pred6) == EXPECTED_TEST_SAMPLES and pred6['id'].is_unique
assert pred6['run_id'].eq(PRIMARY_RUN_ID).all() and pred6['split'].eq('test_official').all()
np.testing.assert_array_equal(pred6['id'].astype(str).to_numpy(), test_ids)
np.testing.assert_array_equal(pred6[[f'true_{l}' for l in LABELS]].to_numpy(), y_test)
scores6 = pred6[[f'score_{l}' for l in LABELS]].to_numpy(dtype=np.float64)
assert np.isfinite(scores6).all() and ((scores6 >= 0) & (scores6 <= 1)).all()
np.testing.assert_allclose(average_precision_score(y_test,scores6,average=None), primary6['Average Precision'], rtol=0, atol=1e-12)
np.testing.assert_array_equal(pred6[[f'pred_optimized_{l}' for l in LABELS]].to_numpy(), (scores6 >= primary6['Threshold'].to_numpy()).astype(np.int8))

# Keep the frozen validation order in all cross-model figures.
short6 = {'S02_ROBERTA_FOCAL_FULL_S42':'S02 RoBERTa Focal (full)', 'T06_FOCAL_S42':'T06 RoBERTa Focal', 'T05_WBCE_S42':'T05 RoBERTa WBCE', 'T04_BCE_S42':'T04 RoBERTa BCE', 'T03_FOCAL_S42':'T03 DistilBERT Focal', 'T02_WBCE_S42':'T02 DistilBERT WBCE', 'T01_BCE_S42':'T01 DistilBERT BCE', 'S01_LR_COMBINED_BALANCED_FULL_S42':'S01 Balanced LR combined (full)'}
def model_label6(rid):
    return short6.get(rid, rid.removesuffix('_S42').replace('_',' '))

figures6 = []
def save_figure6(fig, name):
    path = FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.png')
    fig.savefig(temp, dpi=300, bbox_inches='tight', facecolor='white')
    os.replace(temp, path)
    figures6.append({'path':str(path.relative_to(PROJECT_ROOT)), 'sha256':calculate_sha256(path), 'size_bytes':path.stat().st_size, 'dpi':300})
    plt.show()
    plt.close(fig)

with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'}):
    # F01: validation versus test AP, with test F1 alongside.
    fig, axes = plt.subplots(1,2,figsize=(13,9),sharey=True,layout='constrained')
    y = np.arange(len(comparison6))
    axes[0].barh(y-.17,comparison6['Validation Macro AP'],height=.32,label='Validation',color='#a6b8cb')
    axes[0].barh(y+.17,comparison6['Macro AP'],height=.32,label='Official test',color='#34699a')
    axes[0].set_yticks(y,[model_label6(r) for r in comparison6['Run ID']])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Macro average precision')
    axes[0].legend(loc='lower right')
    axes[1].barh(y,comparison6['Optimized Macro F1'],height=.55,color='#31846b',label='Frozen per-label thresholds')
    axes[1].set_xlabel('Official test macro F1')
    for ax in axes:
        ax.set_xlim(0,1); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Model performance in frozen validation rank order')
    save_figure6(fig,'F01_test_model_comparison.png')

    # F02: primary model label-level operating performance.
    fig, ax = plt.subplots(figsize=(10,4.6),layout='constrained')
    x = np.arange(len(LABELS))
    for j,(metric,color) in enumerate([('Precision','#34699a'),('Recall','#e09a46'),('F1','#31846b')]):
        bars = ax.bar(x+(j-1)*.25,primary6[metric],width=.24,label=metric,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
    ax.set_xticks(x,[l.replace('_','\n') for l in LABELS]); ax.set_ylim(0,1.08)
    ax.set_ylabel('Score'); ax.legend(ncol=3,loc='upper center'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.set_title('Frozen primary model: official test precision, recall and F1')
    save_figure6(fig,'F02_final_label_precision_recall_F1.png')

    # F03: six binary confusion matrices. Normalize by actual class for visibility.
    fig, axes = plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,label in zip(axes.flat,LABELS):
        row = primary6.loc[label]
        counts = np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]],dtype=int)
        assert counts.sum() == EXPECTED_TEST_SAMPLES
        rates = counts / counts.sum(axis=1,keepdims=True)
        im = ax.imshow(rates,cmap='Blues',vmin=0,vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{counts[i,j]:,}\n{rates[i,j]:.1%}',ha='center',va='center',color='white' if rates[i,j]>.55 else '#17283b',fontsize=10)
        ax.set_xticks([0,1],['Negative','Positive']); ax.set_yticks([0,1],['Negative','Positive'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(label.replace('_',' '))
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Fraction within actual class')
    fig.suptitle('Frozen primary model: binary confusion matrices (counts and row percentages)')
    save_figure6(fig,'F03_final_binary_confusion_matrices.png')

    # F04: exact PR curves; mark the frozen cutoff, not the test-optimal cutoff.
    fig, axes = plt.subplots(2,3,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
    for i,(ax,label) in enumerate(zip(axes.flat,LABELS)):
        p,r,_ = precision_recall_curve(y_test[:,i],scores6[:,i])
        row = primary6.loc[label]
        ax.step(r,p,where='post',color='#34699a',label=f"AP = {row['Average Precision']:.3f}")
        ax.axhline(y_test[:,i].mean(),color='#888888',linestyle='--',linewidth=1,label='Prevalence')
        ax.scatter(row['Recall'],row['Precision'],color='#c67528',s=30,zorder=5,label='Frozen cutoff')
        ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.set_title(label.replace('_',' '))
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.grid(alpha=.15); ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Frozen primary model: official test precision-recall curves')
    save_figure6(fig,'F04_final_PR_curves.png')

    # F05: default versus validation-tuned thresholds for all configurations.
    fig, ax = plt.subplots(figsize=(10,9),layout='constrained')
    ax.barh(y-.17,comparison6['Default Macro F1'],height=.32,label='Default cutoff',color='#a6b8cb')
    ax.barh(y+.17,comparison6['Optimized Macro F1'],height=.32,label='Validation-tuned cutoffs',color='#31846b')
    ax.set_yticks(y,[model_label6(r) for r in comparison6['Run ID']]); ax.invert_yaxis()
    ax.set_xlim(0,1); ax.set_xlabel('Official test macro F1'); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    ax.legend(loc='lower right'); ax.set_title('Threshold transfer to official test (frozen validation order)')
    save_figure6(fig,'F05_default_vs_tuned_test.png')

# Save matched research comparisons with explicit run IDs.
matched6 = [
    ('Full-data scaling: LR','LR_COMBINED_BALANCED_S42','S01_LR_COMBINED_BALANCED_FULL_S42'),
    ('Full-data scaling: RoBERTa','T06_FOCAL_S42','S02_ROBERTA_FOCAL_FULL_S42'),
    ('DistilBERT: WBCE vs BCE','T01_BCE_S42','T02_WBCE_S42'),
    ('DistilBERT: focal vs BCE','T01_BCE_S42','T03_FOCAL_S42'),
    ('RoBERTa: WBCE vs BCE','T04_BCE_S42','T05_WBCE_S42'),
    ('RoBERTa: focal vs BCE','T04_BCE_S42','T06_FOCAL_S42'),
    ('Backbone: BCE','T01_BCE_S42','T04_BCE_S42'),
    ('Backbone: WBCE','T02_WBCE_S42','T05_WBCE_S42'),
    ('Backbone: focal','T03_FOCAL_S42','T06_FOCAL_S42')]
lookup6 = comparison6.set_index('Run ID')
rows6 = []
for title,ref,candidate in matched6:
    a,b = lookup6.loc[ref],lookup6.loc[candidate]
    rows6.append({'Comparison':title,'Reference Run':ref,'Candidate Run':candidate,
                  'Reference Training Samples':int(a['Training Samples']),'Candidate Training Samples':int(b['Training Samples']),
                  'Reference Test Macro AP':a['Macro AP'],'Candidate Test Macro AP':b['Macro AP'],'Macro AP Change':b['Macro AP']-a['Macro AP'],
                  'Reference Test Macro F1':a['Optimized Macro F1'],'Candidate Test Macro F1':b['Optimized Macro F1'],'Macro F1 Change':b['Optimized Macro F1']-a['Optimized Macro F1'],
                  'Interpretation Scope':'Data and optimizer-update budget jointly differ' if title.startswith('Full-data') else 'Matched representative subset; single seed'})
matched_df6 = pd.DataFrame(rows6)
matched_path6 = TABLES_DIR/'test_matched_research_comparisons.csv'
save_csv_atomic(matched_path6,matched_df6)
assert len(pd.read_csv(matched_path6)) == len(matched6)
assert len(figures6) == 5
assert calculate_sha256(FROZEN_MANIFEST_PATH) == EXPECTED_FROZEN_HASH
assert calculate_sha256(REGISTRY_PATH) == frozen_manifest['registry']['sha256']
assert calculate_sha256(DECISION_LOG_PATH) == frozen_manifest['decision_log_sha256']
assert calculate_sha256(metrics_path6) == metrics_hash6
figure_summary6 = {'status':'Completed','frozen_manifest_sha256':EXPECTED_FROZEN_HASH,'test_dataset_sha256':EXPECTED_TEST_HASH,
                   'primary_selected_run_id':PRIMARY_RUN_ID,'metrics_summary_sha256':calculate_sha256(metrics_path6),
                   'primary_predictions_sha256':str(r6['SHA256']),'figures':figures6,
                   'matched_comparison_table':{'path':str(matched_path6.relative_to(PROJECT_ROOT)),'sha256':calculate_sha256(matched_path6),'rows':len(matched6)},
                   'thresholds_changed':False,'test_driven_selection_performed':False,'additional_inference_performed':False,'model_training_performed':False}
save_json_atomic(LOGS_DIR/'test_figures_summary.json',figure_summary6)
print('\nMatched test comparisons (candidate minus reference)')
display(matched_df6[['Comparison','Reference Run','Candidate Run','Macro AP Change','Macro F1 Change']].round(6))
del pred6,scores6
gc.collect()
print('Verified figures:',len(figures6),'(300 dpi PNG)')
print('Matched comparison records:',len(matched_df6))
print('Thresholds and frozen manifest: Unchanged')
print('Additional inference and training: None')
print('Cell 6 completed successfully.')
