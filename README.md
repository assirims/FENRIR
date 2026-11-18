# FENRIR — Federated Learning-Driven Mathematical Analysis of Cyberthreat Detection in Edge IIoT Networks Using Hybrid Optimization Algorithm

Project: PSAU Project 2025/01/33078

## Folder contents

| Folder | Contents |
|---|---|
| `01_Manuscript` | `FENRIR_Manuscript.docx` (editable, native Word equations), `FENRIR_Manuscript.pdf` (typeset with XeLaTeX), cover letter, highlights, and a zip with the LaTeX/Markdown source + build script |
| `02_Figures` | All 14 figures at 300 dpi (architecture, flowchart, TAG-GNN diagram, data, optimiser, FS, HPO, training curves, confusion matrix, per-class, robustness, non-IID, interpretability, efficiency) |
| `03_Code` | Complete implementation (pure NumPy / SciPy / pandas / scikit-learn; no GPU or deep-learning framework needed) |
| `04_Data` | Raw RT-IoT2022 (original UCI zip), processed standardised data (all 92 features, zipped; and the 24 selected features as CSV) with label, split and client columns, federated correlation matrix, client partition counts, selected-feature list |
| `05_Results` | Summary JSONs (benchmarks, feature selection, tuning, cost, interpretability, all paper tables) and a zip with every run, log, trained FENRIR weights and test-set probabilities |

## Method in one paragraph

FENRIR = (Phase I) secure-sum federated preprocessing and a correlation prior graph → (Phase II) **LC-GW2O**, a Lévy–Chaotic Grey-Wolf–Whale hybrid optimiser used for privacy-preserving federated feature selection (clients return only scalar fitness values) and federated hyper-parameter tuning → (Phase III) **TAG-GNN**, a Tri-Attention Gated Graph Neural Network (feature-as-node graph, periodic tokenizer, prior + adaptive adjacency, graph branch + structure-biased multi-head attention + channel attention with learned gating, attention pooling and dual-view readout), trained with proximal local updates and **MATA** (Median-Anchored Trust-weighted Aggregation) for Byzantine robustness.

## Code map (`03_Code`)

| File | Purpose |
|---|---|
| `autograd.py` | Reverse-mode automatic differentiation engine (NumPy) |
| `models.py` | TAG-GNN (proposed) and baselines MLP, CNN1D, GCN, GAT, FT-Transformer |
| `optimizers.py` | LC-GW2O (proposed) + PSO, DE, GWO, WOA, HHO; shifted benchmark suite |
| `data.py` | RT-IoT2022 / Edge-IIoTset loaders, federated statistics, Dirichlet non-IID partition, prior graph |
| `fs.py` | Federated wrapper feature-selection fitness |
| `fl.py` | Federated training, aggregators (FedAvg, Median, Trimmed mean, Krum, MATA), attacks, metrics |
| `exp_bench.py` | Exp. 1 – optimiser benchmarks + component ablation |
| `exp_fs.py` | Exp. 2 – federated feature selection (run `python exp_fs.py p1 LCGW2O,PSO,WOA`, `p2 GWO,DE,HHO`, then `merge`) |
| `exp_hpo.py` | Exp. 3 – federated hyper-parameter optimisation (`LCGW2O` or `RS`) |
| `exp_fl.py` | Exp. 4–8 – main comparison, ablation, robustness, non-IID, centralised/local references |
| `exp_complexity.py` | Exp. 9 – parameters, MACs, latency, communication |
| `interp.py` | Interpretability of the trained model |
| `diagrams.py`, `make_figures.py` | All figures |
| `fill_results.py` | Builds every table/number of the paper from the JSON results |
| `run_all.sh` | Runs Exp. 4–8 on worker i of n (resumable) |
| `test_grad.py` | Finite-difference gradient check of every model |

## Reproduce

```
pip install numpy scipy pandas scikit-learn matplotlib python-docx
cd 03_Code
python test_grad.py
python exp_bench.py
python exp_fs.py p1 LCGW2O,PSO,WOA ; python exp_fs.py p2 GWO,DE,HHO ; python exp_fs.py merge
python exp_hpo.py LCGW2O ; python exp_hpo.py RS
bash run_all.sh 0 1
python exp_complexity.py ; python interp.py ; python diagrams.py ; python make_figures.py ; python fill_results.py
```
Data path: the loader looks for `../data/RT_IOT2022` and then reads `../04_Data/RT_IOT2022_raw_UCI.zip` directly, so it runs from this folder without unzipping. Scripts write to `../results` and `../figures`; the archived outputs of this study are in `05_Results` (unzip `All_Raw_Results_Runs_Logs_Weights.zip` to get the `results/` folder used by `fill_results.py` and `make_figures.py`). To run on Edge-IIoTset, call `data.prepare(dataset="edge-iiotset", path=".../ML-EdgeIIoT-dataset.csv")` (loader provided; not executed in this study).

## Notes
- All numbers in the paper are produced by these scripts on a 2-core CPU; results are averaged over seeds 0–2 unless stated.
- `05_Results/archive_hpo_v1` keeps the discarded first tuning run (class-weight exponent tuned), discussed in Section 8 of the paper.
