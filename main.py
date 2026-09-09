from torchvision.models import resnet50, resnet18, efficientnet_b0, mobilenet_v3_small
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import mvtec
import os
import sys
import random
import numpy as np

from classes import Teacher, ProjectorWrapper, ReverseDistillationStudent
from actions import train, test, get_validation_threshold
from parameters import RETURN_NODES, SEED, MEAN, STD
from benchmark import benchmark_epochs


def anomaly_detection(dataset_name, teacher_model_name, student_model_name, train_loader, val_loader, test_loader, benchmark_mode=False, max_epochs=100, n_checkpoints=5):
    # INIZIALIZZAZIONE DEVICE
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # INIZIALIZZAZIONE TEACHER
    print(f"Inizializzazione teacher: {teacher_model_name}")
    teacher = Teacher(model_name=teacher_model_name)

    student_model = sys.argv[2].strip()

    students_channels = {
        "resnet50":         {'feat1': 256,  'feat2': 512,   'feat3': 1024},
        "resnet18":         {'feat1': 64,   'feat2': 128,   'feat3': 256},
        "efficientnet":     {'feat1': 24,   'feat2': 80,    'feat3': 192},
        "mobilenet":        {'feat1': 24,   'feat2': 48,    'feat3': 96}
        # altri modelli?
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

    elif student_model_name == "efficientnet":
        student_efficient = efficientnet_b0(weights=None)
        student_layers = create_feature_extractor(student_efficient, return_nodes={'features.2': 'feat1', 'features.4': 'feat2', 'features.6': 'feat3'})
        student_extractor = ProjectorWrapper(student_layers, students_channels["efficientnet"], teacher.channels).to(device)
        reverse_distillation = False

    elif student_model_name == "mobilenet":
        student_mobile = mobilenet_v3_small(weights=None)
        student_layers = create_feature_extractor(student_mobile, return_nodes={'features.2': 'feat1', 'features.8': 'feat2', 'features.11': 'feat3'})
        student_extractor = ProjectorWrapper(student_layers, students_channels["mobilenet"], teacher.channels).to(device)
        reverse_distillation = False

    else:
        print("Errore negli argomenti del programma!")
        exit(1)

    # ADDESTRAMENTO STUDENT (DISTILLAZIONE)
    if not benchmark_mode:
        # MODALITA' STANDARD
        trained_student, _ = train(teacher=teacher.model, 
                            student=student_extractor, 
                            train_loader=train_loader, 
                            epochs=max_epochs, # overfitting per rd4ad?
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

        return accuracy, scores, targets

    else:
        # MODALITA' BENCHMARK
        best_record, history = benchmark_epochs(teacher=teacher.model,
                            student=student_extractor,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            max_epochs=max_epochs,
                            learning_rate=0.001,
                            T=2,
                            device=device,
                            reverse_distillation=reverse_distillation,
                            n_benchmarks=n_checkpoints)

        trainable_params = sum(p.numel() for p in student_extractor.parameters() if p.requires_grad)
        teacher_params = sum(p.numel() for p in teacher.model.parameters())
        size_mb = (trainable_params * 4) / (1024 ** 2)

        return best_record, history, [trainable_params, teacher_params, size_mb]


if __name__ == "__main__":
    # LETTURA PARAMETRI
    benchmark_mode = False

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(f"Uso corretto:\npython3 {sys.argv[0]} <categoria_dataset> <modello_teacher> <modello_student>\npython3 {sys.argv[0]} <categoria_dataset> <modello_teacher> <modello_student> --benchmark\n")
        print("Esempio:\npython main.py bottle wideresnet50 rd4ad")
        exit(1)
    dataset_name = sys.argv[1].strip()
    teacher_model_name = sys.argv[2].strip()
    student_model_name = sys.argv[3].strip()

    if len(sys.argv) == 5 and (sys.argv[4] == "-b" or sys.argv[4] == "--benchmark") :
        benchmark_mode = True


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
        transforms.RandomRotation(degrees=(0, 360)),    # sempre le dannate bottiglie
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

    _,_,_ = anomaly_detection(dataset_name, teacher_model_name, student_model_name, train_loader, val_loader, test_loader, benchmark_mode)

    exit(0)

# TODO: testare tanti students e teachers diversi
# TODO: numero di parametri -> sum(p.numel() for p in student.parameters() if p.requires_grad)
# TODO: misurare tempo per inferenza
# TODO: epoche dinamiche