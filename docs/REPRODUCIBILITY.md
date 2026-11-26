# Reproduction and team handoff

1. Obtain authorized access to the original project workspace and the Jigsaw dataset. Keep raw comments and trained checkpoints outside Git.
2. The Colab notebooks expect `/content/drive/MyDrive/NLP04_Toxic_Comment_Classification`. Restore the exact external artifacts at their recorded paths. Check hashes against the frozen and completion manifests.
3. Use the original workspace for checkpoint replay. Some historical setup/training cells expect the registry at the state of their original stage; do not rerun them against the final registry. To reproduce training from scratch, use a fresh workspace and recreate stages in order. Existing scripts are retained for review, not advertised as an automatically replayable end-to-end pipeline.
4. Notebook 07 Cells 1–8 use the frozen configurations; do not retune against test results. GPU is required for Cell 4. Completed predictions can be reused.
5. Actual recorded evaluation environment: Python 3.13.18, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, PyTorch 2.14.1, Transformers 5.22.1, PyArrow 23.0.1; Tesla T4 CUDA inference used PyTorch 2.14.1+cu130. This is an observed environment, not a tested cross-platform lockfile. Other dependencies are defined in the original notebooks; use the relevant Colab runtime.
6. Pending notebooks 08–11 are not included in this milestone. The frozen primary model and test scores must not be changed based on post-test analysis.

## Publication scope

Original result paths and SHA-256 hashes are recorded in the publication inventory. The frozen and completion manifests refer to external files as well as included files; missing external weights, caches and per-comment predictions are intentional. Notebook outputs and runtime metadata were stripped; obsolete failed upload cells were excluded where identified. Source scripts are not rewritten to alter experimental methodology. The export scans for common credential patterns and row-level CSV headers, but a human review is required before publishing.
