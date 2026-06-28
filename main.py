from torchvision.models import resnet50, ResNet50_Weights, resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import mvtec
import os
import sys
from classes import Teacher, ProjectorWrapper, ReverseDistillationStudent
from actions import train, test, get_validation_threshold
import random
import numpy as np
from parameters import RETURN_NODES, SEED, MEAN, STD

# LETTURA PARAMETRI
if len(sys.argv) < 3:
    print(f"Uso corretto: python3 {sys.argv[0]} <categoria_dataset> <modello_teacher> <modello_student>")
    print("Esempio: python main.py bottle wideresnet50 rd4ad")
    exit(1)
dataset_name = sys.argv[1].strip()
teacher_model_name = sys.argv[2].strip()
student_model_name = sys.argv[3].strip()

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(SEED)

# INIZIALIZZAZIONE DATABASE
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(),    # per textures
    # transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),   # per le bottiglie porca miseria
    transforms.Normalize(mean=MEAN, std=STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

data_train_full = mvtec.MVTEC(root='./mvtec',
                    train=True,
                    transform=transform,
                    target_transform=None,
                    resize=224,
                    interpolation=3,
                    category=dataset_name)

val_size = int(len(data_train_full) * 0.15)
train_size = len(data_train_full) - val_size

data_train, data_val = torch.utils.data.random_split(data_train_full, [train_size, val_size], generator=g)

train_loader = DataLoader(data_train, batch_size=20, shuffle=True, generator=g)
val_loader = DataLoader(data_val, batch_size=20, shuffle=False)

data_test = mvtec.MVTEC(root='./mvtec',
                    train=False,
                    transform=test_transform,
                    target_transform=None,
                    resize=224,
                    interpolation=3,
                    category=dataset_name)

test_loader = DataLoader(dataset=data_test, batch_size=20, shuffle=False)

# INIZIALIZZAZIONE TEACHER
print(f"Inizializzazione teacher: {teacher_model_name}")
teacher = Teacher(model_name=teacher_model_name)

student_model = sys.argv[2].strip()

students_channels = {
    "resnet50": {'feat1': 256, 'feat2': 512, 'feat3': 1024},
    "resnet18": {'feat1': 64, 'feat2': 128, 'feat3': 256},
    # mobilenet, efficientnet ...
}

# INIZIALIZZAZIONE STUDENT
print(f"Inizializzazione student: {student_model_name}")

if student_model_name == "resnet50":
    student_50 = resnet50(weights=None)
    student_layers = create_feature_extractor(student_50, return_nodes=RETURN_NODES)
    student_extractor = ProjectorWrapper(student_layers, students_channels["resnet50"], teacher.channels).to(device)
    reverse_distillation = False

elif student_model_name == "resnet18":
    student_18 = resnet18(weights=None)
    student_layers = create_feature_extractor(student_18, return_nodes=RETURN_NODES)
    student_extractor = ProjectorWrapper(student_layers, students_channels["resnet18"], teacher.channels).to(device)
    reverse_distillation = False

elif student_model_name == "rd4ad":
    student_extractor = ReverseDistillationStudent(teacher_channels=teacher.channels).to(device)
    reverse_distillation = True

else:
    print("Errore negli argomenti del programma!")
    exit(1)

# ADDESTRAMENTO STUDENT (DISTILLAZIONE)
trained_student = train(teacher=teacher.model, 
                    student=student_extractor, 
                    train_loader=train_loader, 
                    epochs=100, 
                    learning_rate=0.001, 
                    T=2, 
                    device=device,
                    reverse_distillation=reverse_distillation)

# CALCOLO THRESHOLD
threshold = get_validation_threshold(teacher=teacher.model,
                                    student=trained_student,
                                    val_loader=val_loader,
                                    device=device,
                                    reverse_distillation=reverse_distillation)

# TEST
accuracy, scores, targets = test(teacher=teacher.model, 
                            student=trained_student, 
                            test_loader=test_loader, 
                            device=device,
                            threshold=threshold,
                            reverse_distillation=reverse_distillation)

exit(0)

# TODO: testare tanti students e teachers diversi
# TODO: numero di parametri -> sum(p.numel() for p in student.parameters() if p.requires_grad)
# TODO: misurare tempo per inferenza
# TODO: epoche dinamiche