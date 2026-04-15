# Mining Domain-Based Policies from Massive and Noisy Access Logs

This repository contains the implementation and experimental evaluation of a high-performance framework for mining domain-based access control policies from large-scale and potentially noisy access logs.

[In previous work](https://doi.org/10.1145/3626232.3653265), Domain-Based Policy Mining (DBPM), often based on [MaxSAT solving](https://github.com/sizhang-ucalgary/DBPM_Experiment), struggle with scalability. This project introduces two novel, GPU-accelerated pipelines that achieve orders of magnitude speedups while maintaining or improving prediction accuracy.

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
│   ├── mdl_compressor.py                       # GPU-accelerated MDL miner
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
    ├── CODASPY2024_DBPM_Dataset/       # Primary benchmarking dataset
    ├── Slurm/                          # SLURM scripts for HPC job submission
    ├── config_dataset.json             # Configuration for dataset generation
    ├── dataset_generator.py            # Script to generate benchmarking instances
    ├── graph_coloring.py               # CPU-based heuristic GC implementations
    ├── maxsat_solver.py                # MaxSAT-based baseline implementation
    ├── test_driver.py                  # Experimental comparison driver
    └── rawdata_analyzer.py             # Script to aggregate results into CSV
```

---

## 🛠️ Getting Started

### Prerequisites

-   **Hardware**: 
    -   Minimum: x86-64 compatible CPU (Intel/AMD) and CUDA-capable GPU (Compute Capability 7.0+ recommended).
    -   Recommended: NVIDIA GPUs with at least 80 GB VRAM (e.g., H100) for massive-scale instances, and 256 GB system memory for CPU-intensive benchmarks.
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
-   **Hardware**: Standard x86-64 compatible CPU (e.g., Intel or AMD) with at least 256 GB of memory recommended.
-   **Usage**: Benchmarking Sequential Graph Coloring and MaxSAT pipeline (§5.3). 

### 2. GPU Acceleration Environment
-   **Hardware**: Standard x86-64 compatible CPU with at least 64 GB host memory and NVIDIA GPUs (the evaluations in the paper utilized H100 SXM with 80 GB VRAM).
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
Benchmarking the CPU-based Graph Coloring heuristics against the MaxSAT solver.

- **Generation of Benchmarking Instances**:
    ```bash
    # Usage: python3.10 dataset_generator.py config_dataset.json
    # Edit config_dataset.json to specify K, M, N, and PS
    cd GC_vs_MaxSAT
    python3.10 dataset_generator.py config_dataset.json
    ```
- **Execution via Test Driver**:
    ```bash
    # Usage: python3.10 test_driver.py solver_type method input_dir output_dir timeout
    #
    # solver_type: maxsat, sergcp
    # method (MaxSAT): BE, BE_CC, BE_NF, BE_NF_LI, BE_NF_FM, BE_NF_MD, BE_NF_MD_LI
    # method (GC):     RS, LF, SL, RSI, LFI, SLI, CSB, CSD, SLF, GIS
    python3.10 test_driver.py sergcp LF CODASPY2024_DBPM_Dataset/M10N100 raw_data 300
    ```
- **Results Aggregation**:
    ```bash
    # Usage: python3 rawdata_analyzer.py <input_dir> <output_dir> <timeout> <timelimit>
    python3 rawdata_analyzer.py raw_data output 300 86400
    ```

#### 4. Batch Experiments (HPC)
For large-scale sweeps (scalability, noise levels, skewness), use the provided automation scripts:
-   **Bash Scripts** (`GC_and_MDL_GPU/Bash/`):
    -   **Purpose**: Automates the end-to-end evaluation of the GPU-accelerated miners.
    -   **Workflow**:
        1.  `Generate_*.sh`: Invokes `policy_generator.py` to create batches of `.npy` datasets.
        2.  `Run_*.sh`: Iterates through datasets, executes miners, and parses output into `.csv` files.
-   **Benchmarking via SLURM** (`GC_vs_MaxSAT/Slurm/`):
    -   **Purpose**: Reproduces the comparative analysis for Figure 2 using HPC job scheduling.
    -   **Execution**: Each subdirectory contains a SLURM script (e.g., `MaxSAT.sh`, `LF.sh`) to execute the `test_driver.py` for a specific configuration across all dataset tiers.
    -   **Expected Output**: Method-specific raw results in JSON format, which are then processed by `rawdata_analyzer.py` to generate cactus plot data.
---

## 📖 Citation
If you use this work in your research, please cite:
```bibtex
@inproceedings{Zhang2026,
  author       = {Si Zhang and Philip W. L. Fong},
  title        = {Mining Domain-Based Policies from Massive and Noisy Access Logs},
  booktitle    = {Proceedings of {SACMAT}},
  publisher    = {{ACM}},
  year         = {2026}
}
```
