
# BINDE: Biologically-Informed Neural Differential Engine

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Research_Release-orange)

**BINDE** is a Grey-Box hybrid computational architecture that bridges the gap between mechanistic biological rigor and machine learning flexibility. By constraining continuous Neural Ordinary Differential Equations (Neural ODEs) with established biological topology (via the Hadamard product), BINDE overcomes the overfitting limitations of pure black-box models in high-dimensional, noisy transcriptomic data.

This repository contains the source code, data engineering pipelines, and the case study reproducing the counterfactual analysis of the **TP53 signaling pathway** in oncological contexts.


## 📌 Key Features

*   **Topologically-Constrained Neural ODEs:** Uses KEGG-derived adjacency matrices as a rigid structural mask ($\mathbf{W} \odot \mathbf{A}$) to filter experimental noise and prevent spurious correlation learning.
*   **Continuous Temporal Forecasting:** Replaces discrete state transitions with a continuous vector field, accurately modeling non-linear biological feedback loops.
*   **Virtual Twins & Counterfactual Analysis:** Natively supports *in silico* genomic perturbations (e.g., node knockdowns) to simulate downstream signal attenuation and dynamic homeostasis.
*   **Scalable & Reproducible Pipeline:** Built with automated ETL harmonization for multi-omics data, ensuring strict tensor alignment between empirical matrices and biological graphs.

---

## 🔬 Paper & Citation

This framework is part of the research submitted to the **Brazilian Symposium on Bioinformatics (BSB) 2026**. 

If you use this framework, please cite:

Guateque Jaramillo, Maria Leandra. "Causal Modeling and Multi-Omics Integration in Biological Systems". 2026. Thesis (PhD in Informatics). Pontifical Catholic University of Rio de Janeiro (PUC-Rio).

---

## ⚙️ Installation & Setup

### Clone the repository

```bash
git clone https://github.com/mjaramillo16/BINDE-Biologically-Informed-Neural-Differential-Engine
cd BINDE
```

### Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> Ensure `gseapy` is installed for automated Gene Ontology annotation.

---

## Reproduction Pipeline 

To ensure full reproducibility of the results, use the global orchestrator.

### Option 1: One-Click Global Orchestrator (Recommended)

```bash
python3 reset_and_run.py
```

This will automatically:

- Clean environment  
- Download raw datasets  
- Build KEGG biological graphs  
- Train all Neural ODE models  
- Execute benchmarking suite  
- Simulate Virtual Twins  
- Generate publication-ready plots  

Outputs are saved in:

```
results/results_[TIMESTAMP]/
```

---

### Option 2: Step-by-Step Execution

#### Stage 1 — Model Fabrication (ETL & Training)

```bash
python3 experiments/run_case_a_cancer.py

```
#### Stage 2. Simulate Virtual Twin (Counterfactual XAI)

Perform an *in silico* knockdown of the TP53 node to evaluate the continuous downstream attenuation of effector genes (e.g., CDKN1A/p21).

```bash
python3 experiments/Test_Gemeos_real_data.py
```

#### Stage 3 — Global Evaluation 

```bash
python3 experiments/run_benchmark_all.py
```

This stage:

- Evaluates trained models  
- Generates Rolling CV metrics  
- Performs in silico causal knockdowns  
- Renders annotated heatmaps (Louvain + GSEA)

---


**Core Dependencies:**

* `torch` and `torchdiffeq` (for the Neural ODE solver and adjoint sensitivity method)
* `bioservices` (for automated KEGG KGML parsing)
* `networkx` & `python-louvain` (for topological diagnostics)
* `pandas`, `numpy`, `scikit-learn` (for ETL and data normalization)

---


## ✉️ Contact & Affiliation

Developed at the **Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio)**.

For questions, collaborations, or access to extended multi-omics modules, please open an issue in this repository or contact the authors.

```
