from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import mvtec
import os
import sys
from teacher import Teacher
from actions import train, test
import random
import numpy as np
from parameters import RETURN_NODES, THRESHOLD, SEED, MEAN, STD

dataset_name = sys.argv[1].strip()

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

g = torch.Generator()
g.manual_seed(SEED)

# caricamento dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=MEAN, std=STD)
])

data_train = mvtec.MVTEC(root='./mvtec',
                    train=True,
                    transform=transform,
                    target_transform=None,
                    resize=224,
                    interpolation=3,
                    category=dataset_name)

data_test = mvtec.MVTEC(root='./mvtec',
                    train=False,
                    transform=transform,
                    target_transform=None,
                    resize=224,
                    interpolation=3,
                    category=dataset_name)

train_loader = DataLoader(dataset=data_train, batch_size=20, shuffle=True, generator=g)
test_loader = DataLoader(dataset=data_test, batch_size=20, shuffle=False)

teacher = Teacher(model="resnet")

# provo come prima cosa lo stesso modello senza pesi
student = resnet50(weights=None)

trained_student = train(teacher=teacher.model, 
                    student=student, 
                    train_loader=train_loader, 
                    epochs=100, 
                    learning_rate=0.001, 
                    T=2, 
                    device=device)

accuracy, scores, targets = test(teacher=teacher.model, 
                            student=trained_student, 
                            test_loader=test_loader, 
                            device=device)

#TODO: creare maschere binarie
#TODO: test con mie maschere binarie vs ground truth