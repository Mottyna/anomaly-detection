# Anomaly Detection
Project work for the Computer Vision course at [Unife](https://corsi.unife.it/it/lm-ia).

The goal of this project was to implement and test the *knowledge distillation* technique for *anomaly detection* across various models on the [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) dataset.

## Main features
* **Multi-architecture:** support for several pre-trained Teacher models (`resnet18`, `resnet50`, `wideresnet50`, `efficientnet`).
* **Two distillation paradigms:**
    * **Standard knowledge distillation:** direct feature alignment using 1x1 convolutions (`ProjectorWrapper`).
    * **Reverse Distillation for Anomaly Detection (RD4AD):** passing Teacher features through a bottleneck (`Bottleneck`), followed by reconstruction using decoders with non-corresponding skip-connections.
* **Automatic threshold calibration:** calculation of the optimal decision threshold on the validation set via synthetic anomaly generation (*Cut-paste* and *Alpha blending noise/color* techniques) and the Youden index.
* **Evaluation:** calculation of performance metrics at both image and pixel levels:
    * image-level & pixel-level ROC-AUC
    * global PR-AUC
    * Precision, Recall, and F1-Score
* **Anomaly visualization:** generation of comparative heatmaps (original image, *ground truth* mask, and *anomaly map*) saved in PNG format.

## Project structure

```text
├── main.py                 # Definition of the `anomaly_detection` function, which trains and tests the teacher-student pair on the anomaly detection task
├── classes.py              # Definition of the networks (Teacher, Projector, RD4AD Student)
├── actions.py              # Core functions: train, test, threshold calibration, and synthetic anomaly generation
├── mvtec.py                # Dataloader for managing Train and Test/GT streams
├── parameters.py           # Constants, seeds, hyperparameters, and layer weights
├── benchmark.py            # Definition of the `benchmark_epochs` function, which trains the student and provides checkpoints with intermediate results
├── run_benchmark.py        # Benchmark of all networks as students and teachers, with checkpoints at different training stages
├── risultati/              # [Auto-generated] Folder containing the test output plots
├── mvtec/                  # Folder that must contain the mvtec databases you want to test the program on
└── risultati_benchmark/    # [Auto-generated] Folder containing the benchmark results
```

## Requirements and installation
The project requires Python 3.8+ and the following libraries:
```bash
pip install torch torchvision scikit-learn numpy matplotlib pillow seaborn
```

### Dataset configuration
Download the MVTec AD dataset and place it inside a folder named `mvtec` in the project's root directory. The folder structure must follow the MVTec standard:

```text
./mvtec/
└── bottle/
    ├── ground_truth/
    |   ├── broken_large/
    |   ├── broken_small/
    |   └── contamination/
    ├── test/
    │   ├── broken_large/
    |   ├── broken_small/
    |   ├── contamination/
    |   └── good/
    └── train/
        └── good/
```

## How to use the project
### Anomaly detection / benchmark of a single teacher-student pair at a time
The `main.py` file accepts three to four positional command-line arguments, depending on the desired mode:
```bash
python3 main.py <dataset_category> <teacher_model> <student_model>
python3 main.py <dataset_category> <teacher_model> <student_model> -b
python3 main.py <dataset_category> <teacher_model> <student_model> --benchmark
```
The presence of `-b` or `--benchmark` indicates the benchmark mode, which involves saving checkpoints during the training phase.

To automatically test all models, use the `run_benchmark.py` file!

### Full benchmark
The `run_benchmark.py` file accepts up to three positional arguments:
```bash
python3 run_benchmark.py <dataset_category> <max_epochs> <n_checkpoints>
```

### Execution examples:
Run with reverse distillation (RD4AD) on "bottle":
```bash
python3 main.py bottle wideresnet50 rd4ad
```
Run with standard distillation on "carpet":
```bash
python3 main.py carpet resnet50 resnet18
```
Run benchmark on "bottle", with a maximum of 150 epochs and 15 checkpoints:
```bash
python3 run_benchmark.py bottle 150 15
```

## Relevant technical details
### Edge effect prevention
During the anomaly map extraction phase (`actions.py`), the image edges tend to generate false positives. The code artificially zeros out a 10-pixel margin along the edges to clean the signal.

### Top-K pixel pooling
To prevent the physical size of the defect from overly influencing the global image score, the image-level metric is calculated by extracting only the worst 0.5% of pixels (those with the highest error) and averaging them.

### Layer balancing (`parameters.py`)
It is possible to assign different weights to the layers of the student network in *reverse distillation* depending on the dataset category (e.g., prioritizing initial layers for textures or deeper layers for shapes).

## Output
### Standard mode (`main.py`)
At the end of the test phase (only if run in standard mode), comparative images named `anomaly_sample_[ID].png` will be saved in the `risultati/` (results) folder. Each image contains:
* **Original**: The denormalized input image with the real target.
* **Ground Truth**: The real binary mask of the defect.
* **Anomaly Map (Heatmap)**: The heat map generated by the model, alongside the system's final prediction (Normal or Anomalous).

The evaluation metrics will be printed to the terminal:
- ROC-AUC (global and pixel-level)
- PR-AUC
- F1 Score
- Precision
- Recall
- Optimal threshold calculated using the test set (*not* the one used to compute the metrics)

### Benchmark mode (`main.py`)
The evaluation metrics will be printed to the terminal at each training checkpoint:
- ROC-AUC (global and pixel-level)
- PR-AUC
- F1 Score
- Precision
- Recall
- Optimal threshold calculated using the test set (*not* the one used to compute the metrics)

### Benchmark (`run_benchmark.py`)
In the `risultati_benchmark` (benchmark results) folder, a `.csv` file containing the best results obtained by each tested teacher-student pair and other metrics, alongside a `.json` file documenting the training progress, will be saved.

## Credits 🙏
This project includes and adapts third-party open-source code:

* **MVTec Dataloader (`mvtec.py`):** Adapted from the original repository by [@b3r8](https://github.com/b3r8) ([b3r8/mvtec-dataloader](https://github.com/b3r8/mvtec-dataloader)), released under the **MIT** license. The original file was modified to integrate and return the Ground Truth (GT) masks required for pixel-level evaluation.

My implementation of reverse distillation is inspired by the idea presented in the paper [SK-RD4AD](https://openaccess.thecvf.com/content/CVPR2025W/VAND/html/Park_SK-RD4AD__Skip-Connected_Reverse_Distillation_For_Robust_One-Class_Anomaly_Detection_CVPRW_2025_paper.html), implemented in the [SK-RD4AD](https://github.com/pej0918/SK-RD4AD/tree/main) repository.