from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import mvtec
import os
import sys
from classes import Teacher, ProjectorWrapper, ReverseDistillationStudent
from actions import train, test
import random
import numpy as np
from parameters import RETURN_NODES, THRESHOLD_RESNET50, THRESHOLD_RESNET18, THRESHOLD_RD4AD, SEED, MEAN, STD

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
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(SEED)

# caricamento dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=MEAN, std=STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
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
                    transform=test_transform,
                    target_transform=None,
                    resize=224,
                    interpolation=3,
                    category=dataset_name)

train_loader = DataLoader(dataset=data_train, batch_size=20, shuffle=True, generator=g)
test_loader = DataLoader(dataset=data_test, batch_size=20, shuffle=False)

teacher = Teacher(model="resnet")

student_model = sys.argv[2].strip()

if student_model == "resnet50":
    # stesso modello senza pesi
    student_50 = resnet50(weights=None)
    student_extractor = create_feature_extractor(student_50, return_nodes=RETURN_NODES)
    threshold = THRESHOLD_RESNET50
    reverse_distillation = False
elif student_model == "resnet18":
    # modello piu' leggero
    student_18 = resnet18(weights=None)
    student_layers = create_feature_extractor(student_18, return_nodes=RETURN_NODES)
    student_extractor = ProjectorWrapper(student_layers).to(device)
    threshold = THRESHOLD_RESNET18
    reverse_distillation = False
elif student_model == "rd4ad":
    student_extractor = ReverseDistillationStudent().to(device)
    threshold = THRESHOLD_RD4AD
    reverse_distillation = True
else:
    print("Errore negli argomenti del programma!")
    exit(1)

trained_student = train(teacher=teacher.model, 
                    student=student_extractor, 
                    train_loader=train_loader, 
                    epochs=100, 
                    learning_rate=0.001, 
                    T=2, 
                    device=device,
                    reverse_distillation=reverse_distillation)

accuracy, scores, targets = test(teacher=teacher.model, 
                            student=trained_student, 
                            test_loader=test_loader, 
                            device=device,
                            threshold=threshold,
                            reverse_distillation=reverse_distillation)

exit(0)

#TODO: testare tanti student diversi e magari RD4AD