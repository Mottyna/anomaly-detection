from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
import torch
from anomalib.data import MVTecAD
import os
import sys
from teacher import Teacher
from actions import train

dataset_name = sys.argv[1].strip()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# TODO: scrivere caricamento dati custom
# caricamento dataset
datamodule = MVTecAD(
    root=f"./datasets/{dataset_name}",
    category=f"{dataset_name}"
)

train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()

teacher = Teacher(model="resnet")

# provo come prima cosa lo stesso modello senza pesi
student = resnet50(weights=None)

train(teacher=teacher.model, student=student, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)

#TODO: test con mie maschere binarie vs ground truth
