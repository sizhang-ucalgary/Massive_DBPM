# Mining Domain-Based Policies from Massive and Noisy Access Logs

This repository contains the implementation and experimental evaluation of a high-performance framework for mining domain-based access control policies from large-scale and potentially noisy access logs.

Traditional Domain-Based Policy Mining (DBPM) approaches, often based on MaxSAT solving, struggle with scalability. This project introduces two novel, GPU-accelerated pipelines that achieve orders of magnitude speedups while maintaining or improving prediction accuracy.

## 🚀 Key Contributions

-   **Graph Coloring-Based Pipeline**: Reduces the clean setting DBPM problem to vertex coloring, leveraging GPU acceleration to handle large-scale access control tensors.
-   **MDL-Based Miner**: Applies the Minimum Description Length (MDL) principle for the noisy setting, acting as a statistical regularizer to recover policy structures from imperfect logs.
-   **Extreme Scalability**: Aggressive GPU-parallelization eliminates computational bottlenecks, outperforming state-of-the-art MaxSAT and Machine Learning (Decision Trees, MLP) baselines.

---

## 📁 Project Structure

```text
.
├── GC_and_MDL_GPU/             # High-performance GPU implementations
│   ├── Bash/                                   # Shell scripts for experiment automation
│   ├── gc_compressor.py                        # GPU-accelerated Graph Coloring pipeline
│   ├── mdl_compressor.py                       # GPU-accelerated MDL (AutoPart-GPU) miner
│   ├── policy_generator.py                     # Synthetic dataset generation engine
│   ├── seandroid_policy_converter.py           # Converter for SEAndroid datasets
│   ├── dt_sklearn.py                           # Decision Tree baseline
│   ├── mlp_torch.py                            # MLP/Deep Learning baseline
│   ├── compare_methods_skewness.py             # Comparison across distribution shifts
│   ├── seandroid_dataset.pkl                   # Processed SEAndroid policy data
│   ├── visual_seandroid_synthetic_tables.py    # Latex table generator
│   └── visual_skewness_tikzpicture.py          # PGFPlots visualization generator
│
└── GC_vs_MaxSAT/               # CPU-based benchmarking suite
    ├── slurm/                          # Slurm workload manager scripts
    ├── input/                          # Sample problem instances
    ├── maxsat_solver.py                # Implementation of previous MaxSAT approach
    ├── maxsat_data_analyzer.py         # Analysis tools for MaxSAT experiments
    ├── graph_coloring.py               # CPU-based heuristic GC alternatives
    ├── gc_heuristic_analyzer.py        # Heuristic comparison analyzer
    ├── test_driver.py                  # Experimental comparison driver
    ├── dataset_generator.py            # Dataset generator for MaxSAT benchmarks
    ├── config_dataset.json             # Configuration for dataset generation
    └── cactus.tex                      # Latex template for cactus plots
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
    -   CPU: `networkx>=3.4.2`, `numba>=0.62.1`, `numpy>=2.2.2`, `scipy>=1.15.1`, `python-sat>=0.1.8.dev20`

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
    # 
    # k: Number of actions
    # m: Initial number of domains
    # n: Number of entities
    # ps: Wildcard probability
    # pn: Noise probability (pn > 0 triggers bit-packed Mode B)
    # --alpha: Dirichlet alpha for domain skewness (default: 1.0)
    # --dir: Output directory (default: 'Policy')
    python GC_and_MDL_GPU/policy_generator.py 1 100 1000 0.3 0.0 --dir PolicyData
    ```
-   **SEAndroid Data**:
    ```bash
    # Usage: python seandroid_policy_converter.py n ps pn input [out_dir]
    # 
    # n: Target number of entities (e.g., 50000)
    # ps: Wildcard probability (e.g., 0.1)
    # pn: Noise probability (e.g., 0.05)
    # input: Path to seandroid_dataset.json or .pkl
    # out_dir: Optional output directory (default: 'Policy')
    python GC_and_MDL_GPU/seandroid_policy_converter.py 3000 0.3 0.1 GC_and_MDL_GPU/seandroid_dataset.pkl PolicyData
    ```

#### 2. Policy Mining
Extract policies from the generated datasets using the GPU-accelerated pipelines.
-   **Clean Setting (Graph Coloring)**:
    ```bash
    # Usage: python gc_compressor.py partial_policy original_policy [--device {cpu,gpu}] [--verbose]
    # 
    # partial_policy: Path to the partial policy (.npy) - standard (int8) format
    # original_policy: Path to the original policy (.npy) for testing
    # --device: Execution mode, either 'cpu' or 'gpu' (default: 'gpu')
    # --verbose: Enable verbose progress output
    python GC_and_MDL_GPU/gc_compressor.py PolicyData/partial_policy_... PolicyData/original_policy_... --device gpu
    ```
-   **Noisy Setting (MDL)**:
    ```bash
    # Usage: python mdl_compressor.py noise_policy original_policy [--device {cpu,gpu}] [--verbose]
    # 
    # noise_policy: Path to the noise policy (.npy) - MUST be bit-packed (uint8)
    # original_policy: Path to the original policy (.npy) for testing
    # --device: Execution mode, either 'cpu' or 'gpu' (default: 'gpu')
    # --verbose: Enable verbose progress output
    python GC_and_MDL_GPU/mdl_compressor.py PolicyData/noise_policy_... PolicyData/original_policy_... --device gpu
    ```

#### 3. MaxSAT vs. GC Comparison
Benchmarking the CPU-based Graph Coloring heuristics against the MaxSAT solver using `.json` instances.
```bash
# Usage: python test_driver.py solver_type method input_dir output_dir timeout
# 
# solver_type: maxsat, sergcp, or pargcp
# method: 
#   - For maxsat:  BE, BE_CC, BE_NF, BE_NF_LI, BE_NF_FM, BE_NF_MD, BE_NF_MD_LI
#   - For sergcp:  RS, LF, SL, RSI, LFI, SLI, CSB, CSD, SLF, GIS
#   - For pargcp:  D1, D1-2GL
#
# Example: Compare RS heuristic against MaxSAT
python GC_vs_MaxSAT/test_driver.py sergcp RS GC_vs_MaxSAT/input/ Results 300
python GC_vs_MaxSAT/test_driver.py maxsat BE_NF_MD_LI GC_vs_MaxSAT/input/ Results 300
```

#### 4. Batch Experiments (HPC)
For large-scale sweeps (scalability, noise levels, skewness), use the provided automation scripts:
-   **Bash Scripts** (`GC_and_MDL_GPU/Bash/`):
    -   **Purpose**: Automates the end-to-end evaluation of the GPU-accelerated miners.
    -   **Workflow**:
        1.  `Generate_*.sh`: Invokes `policy_generator.py` to create batches of `.npy` datasets with varying parameters (e.g., scalability `n`, noise `pn`, or skewness `alpha`).
        2.  `Run_*.sh`: Iterates through the generated datasets, executes the appropriate miner (`gc_compressor.py` or `mdl_compressor.py`), and parses the console output into a consolidated `.csv` file for statistical analysis.
-   **Slurm Scripts** (`GC_vs_MaxSAT/slurm/`):
    -   **Purpose**: Designed for batch execution on a high-performance cluster using the Slurm workload manager.
    -   **Configuration**: Typically requests high-memory partitions (e.g., `--partition=bigmem` with `256gb` RAM) to handle large-scale graph construction for CPU benchmarks.
    -   **Naming Convention**: `[Algorithm]_N[Size].slurm` (e.g., `RS_N1000.slurm`) runs a specific graph coloring heuristic on all problem instances of size $N=1000$.
    -   **Execution**: Invokes `test_driver.py` to compare heuristics against MaxSAT baselines across multiple dataset sub-variants (e.g., varying domain counts $M$) within a single job.
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
