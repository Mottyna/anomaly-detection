import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet18, efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd
import json
import os
import sys
import random
import numpy as np

import mvtec
from benchmark import benchmark_epochs
from parameters import RETURN_NODES, SEED, MEAN, STD
from main import distilla

"""
TEACHERS = ["resnet50", "wideresnet50", "resnet18", "efficientnet"]
STUDENTS = ["rd4ad", "resnet18", "resnet50", "efficientnet"]
"""

TEACHERS = ["resnet50"]
STUDENTS = ["efficientnet"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_full_benchmark(category="bottle", max_epochs=100, n_checkpoints=10, batch_size=20, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Benchmark [{category.upper()}] on {device.type.upper()} ===")

    # SEED
    set_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    # DATABASE
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    full_train = mvtec.MVTEC(root='./mvtec', train=True, transform=train_transform, resize=224, category=category)
    val_size = int(len(full_train) * 0.15)
    train_size = len(full_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_ds = mvtec.MVTEC(root='./mvtec', train=False, transform=test_transform, resize=224, category=category)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # BENCHMARK
    summary_results = []
    all_trajectories = {}

    for t_name in TEACHERS:
        for s_name in STUDENTS:
            pair_name = f"T:{t_name}__S:{s_name}"
            print(f"\n==================================================")
            print(f"  Benchmarking now: {pair_name}")
            print(f"==================================================")

            set_seed(SEED)

            best_record, history, [trainable_params, teacher_params, size_mb] = distilla(dataset_name=category,
                                                                                teacher_model_name=t_name,
                                                                                student_model_name=s_name,
                                                                                train_loader=train_loader,
                                                                                val_loader=val_loader,
                                                                                test_loader=test_loader,
                                                                                benchmark_mode=True,
                                                                                max_epochs=max_epochs,
                                                                                n_checkpoints=n_checkpoints)

            all_trajectories[pair_name] = history

            summary_results.append({
                "Teacher": t_name,
                "Student": s_name,
                "Optimal_Epoch": best_record["epoch"],
                "Time_To_Optimal_Sec": best_record["cumulative_train_sec"],
                "Best_Img_ROC_AUC": round(best_record["image_roc_auc"], 2),
                "Best_Pixel_ROC_AUC": round(best_record["pixel_roc_auc"], 2),
                "Best_F1_Score": round(best_record["f1_score"], 2),
                "Accuracy": round(best_record["accuracy"], 2),
                "PR_AUC": round(best_record["pr_auc"], 2),
                "Latency_ms": round(best_record["inference_latency_ms"], 2),
                "FPS": round(best_record["fps"], 1),
                "Student_Params_M": round(trainable_params / 1e6, 3),
                "Teacher_Params_M": round(teacher_params / 1e6, 3),
                "Student_Size_MB": round(size_mb, 2)
            })

    df_summary = pd.DataFrame(summary_results)

    csv_path = os.path.join(output_dir, f"benchmark_summary_{category}.csv")
    json_path = os.path.join(output_dir, f"epoch_trajectories_{category}.json")
    
    df_summary.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_trajectories, f, indent=4)

    print("\n" + "=" * 90)
    print("                      OPTIMAL BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print(f"\nSummary exported to '{csv_path}'")
    print(f"Detailed epoch trajectories saved to '{json_path}'")


if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_checkpoints = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    run_full_benchmark(category=category, max_epochs=max_epochs, n_checkpoints=n_checkpoints)