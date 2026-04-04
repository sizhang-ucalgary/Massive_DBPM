# Massive DBPM: Mining Domain-Based Policies from Massive and Noisy Access Logs

This repository contains the implementation and experimental evaluation of a high-performance framework for mining domain-based access control policies from large-scale and potentially noisy access logs.

Traditional Domain-Based Policy Mining (DBPM) approaches, often based on MaxSAT solving, struggle with scalability. This project introduces two novel, GPU-accelerated pipelines that achieve orders of magnitude speedups while maintaining or improving prediction accuracy.

## 🚀 Key Contributions

-   **Graph Coloring-Based Pipeline**: Reduces the clean setting DBPM problem to vertex coloring, leveraging GPU acceleration to handle large-scale access control tensors.
-   **MDL-Based Pipeline (AutoPart-GPU)**: Applies the Minimum Description Length (MDL) principle for the noisy setting, acting as a statistical regularizer to recover policy structures from imperfect logs.
-   **Extreme Scalability**: Aggressive GPU-parallelization eliminates computational bottlenecks, outperforming state-of-the-art MaxSAT and Machine Learning (Decision Trees, MLP) baselines.

---

## 📁 Project Structure

```text
.
├── GC_and_MDL_GPU/                     # High-performance GPU implementations
│   ├── Bash/                           # Shell scripts for experiment automation
│   ├── gc_compressor.py                # GPU-accelerated Graph Coloring pipeline
│   ├── mdl_compressor.py               # GPU-accelerated MDL (AutoPart-GPU) miner
│   ├── policy_generator.py             # Synthetic dataset generation engine
│   ├── dt_sklearn.py                   # Decision Tree baseline
│   ├── mlp_torch.py                    # MLP/Deep Learning baseline
│   ├── compare_methods_skewness.py     # Comparative scripts for skewness
│   ├── seandroid_dataset.pkl           # Processed SEAndroid policy data
│   └── visual_*.py                     # Scripts for generating paper visualizations
│
└── GC_vs_MaxSAT/                       # CPU-based benchmarking suite
    ├── slurm/                          # Slurm workload manager scripts
    ├── maxsat_solver.py                # Implementation of previous MaxSAT approach
    ├── graph_coloring.py               # CPU-based heuristic GC alternatives
    ├── test_driver.py                  # RQ1 comparison driver
    └── dataset_generator.py            # Dataset generator for MaxSAT benchmarks
```

---

## 🛠️ Getting Started

### Prerequisites

-   **Hardware**: 
    -   Minimum: CUDA-capable GPU (Compute Capability 7.0+ recommended).
    -   Recommended: NVIDIA H100 SXM (80 GB VRAM) for massive-scale instances.
-   **Software**:
    -   **Python 3.10+**
    -   **CUDA Toolkit 12.9+**
-   **Dependencies**:
    -   GPU: `cupy>=13.6.0`, `torch>=2.10.0`, `scikit-learn>=1.8.0`
    -   CPU: `networkx>=3.4.2`, `numba>=0.62.1`, `numpy>=2.2.2`, `scipy>=1.15.1`

---

## 🏗️ Experiment Setup (HPC)

The evaluations reported in the paper were conducted across two high-performance computing (HPC) environments:

### 1. Baseline CPU Environment
-   **Hardware**: Intel 6148 CPU (2.4 GHz) with 3 TB of system memory.
-   **Usage**: Benchmarking against the MaxSAT pipeline (§5.3). Sequential GC experiments were allocated 256 GB of memory.

### 2. GPU Acceleration Environment
-   **Hardware**: NVIDIA H100 SXM GPUs (80 GB VRAM) interconnected via NVLink, paired with Intel 8570 CPUs (2.1 GHz) and 64 GB host memory.
-   **Usage**: Scalability, noise resilience, and distribution skewness experiments (§5.4, §5.5, §5.6).

---

### Running Experiments

The experimental workflow consists of three main phases: dataset generation, policy mining, and large-scale benchmarking.

#### 1. Dataset Generation
Generate synthetic or SEAndroid-based access control tensors in `.npy` format.
-   **Synthetic Data**:
    ```bash
    # Usage: python policy_generator.py k m n ps pn [--alpha ALPHA] [--dir DIR]
    # pn=0 generates Mode A (int8) data; pn>0 generates Mode B (uint8 bit-packed) data
    python GC_and_MDL_GPU/policy_generator.py 1 100 1000 0.3 0.0 --dir PolicyData
    ```
-   **SEAndroid Data**:
    ```bash
    # Usage: python seandroid_policy_converter.py n ps pn input [out_dir]
    python GC_and_MDL_GPU/seandroid_policy_converter.py 3000 0.3 0.1 GC_and_MDL_GPU/seandroid_dataset.pkl PolicyData
    ```

#### 2. Policy Mining
Extract policies from the generated datasets using the GPU-accelerated pipelines.
-   **Clean Setting (Graph Coloring)**:
    ```bash
    # Requires Mode A (int8) partial_policy file
    python GC_and_MDL_GPU/gc_compressor.py PolicyData/partial_policy_... PolicyData/original_policy_... --device gpu
    ```
-   **Noisy Setting (MDL)**:
    ```bash
    # Requires Mode B (uint8 bit-packed) noise_policy file
    python GC_and_MDL_GPU/mdl_compressor.py PolicyData/noise_policy_... PolicyData/original_policy_... --device gpu
    ```

#### 3. MaxSAT vs. GC Comparison (RQ1)
Benchmarking the CPU-based Graph Coloring heuristics against the MaxSAT solver using `.json` instances.
```bash
# Usage: python test_driver.py solver_type method input_dir output_dir timeout
# Example: Compare RS heuristic against MaxSAT
python GC_vs_MaxSAT/test_driver.py sergcp RS GC_vs_MaxSAT/input/ Results 300
python GC_vs_MaxSAT/test_driver.py maxsat BE_NF_MD_LI GC_vs_MaxSAT/input/ Results 300
```

#### 4. Batch Experiments (HPC)
For large-scale sweeps (scalability, noise levels, skewness), use the provided automation scripts:
-   **Bash Scripts**: Located in `GC_and_MDL_GPU/Bash/` for synthetic and SEAndroid sweeps.
    ```bash
    ./GC_and_MDL_GPU/Bash/Run_Scalability.sh
    ```
-   **Slurm Scripts**: Located in `GC_vs_MaxSAT/slurm/` for cluster job submission.
    ```bash
    sbatch GC_vs_MaxSAT/slurm/RS_N1000.slurm
    ```

---

## 📖 Citation
If you use this work in your research, please cite:
```bibtex
@inproceedings{anonymized2026,
  title={Mining Domain-Based Policies from Massive and Noisy Access Logs},
  author={Anonymized},
  booktitle={ACM Conference},
  year={2026}
}
```
