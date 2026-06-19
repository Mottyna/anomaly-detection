import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from parameters import RETURN_NODES, THRESHOLD

def train(teacher, student, train_loader, epochs, learning_rate, T, device):
    student_extractor = create_feature_extractor(student, return_nodes=RETURN_NODES)

    teacher.eval()
    student_extractor.train()

    teacher = teacher.to(device)
    student_extractor = student_extractor.to(device)

    # criterion = nn.MSELoss()
    optimizer = optim.Adam(student_extractor.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n--- Inizio Addestramento ({epochs} Epoche) ---")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # forward pass con il modello teacher, non salvo gradiente
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            # forward pass con student
            student_outputs = student_extractor(inputs)

            # calcolo della loss tra le mappe di feature nei diversi layers
            loss = 0.0
            for key in RETURN_NODES.values():
                t_feat = teacher_outputs[key]
                s_feat = student_outputs[key]

                # normalizzato
                t_feat_norm = F.normalize(t_feat, p=2, dim=1)
                s_feat_norm = F.normalize(s_feat, p=2, dim=1)

                cos_sim = F.cosine_similarity(s_feat, t_feat, dim=1)
                loss += (1 - cos_sim).mean()

                #loss += criterion(s_feat_norm, t_feat_norm)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()    # aggiorno il learning rate
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    return student_extractor


def test(teacher, student, test_loader, device):
    """
    """
    teacher.eval()
    student.eval()

    total_samples = 0
    correct_predictions = 0
    
    all_anomaly_scores = []
    all_targets = []

    print("\n--- Avvio Fase di Test ---")
    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.to(device)
            inputs = inputs.to(device)
            
            t_outs = teacher(inputs)
            s_outs = student(inputs)
            
            batch_size, _, h, w = inputs.shape

            global_anomaly_map = torch.zeros((batch_size, 1, h, w), device=device)

            for key in RETURN_NODES.values():
                t_feat = F.normalize(t_outs[key], p=2, dim=1)
                s_feat = F.normalize(s_outs[key], p=2, dim=1)

                # anomaly map locale
                # layer_map = torch.mean((t_feat - s_feat) ** 2, dim=1, keepdim=True)

                cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1).unsqueeze(1)
                layer_map = 1 - cos_sim

                # resize
                layer_map_resized = F.interpolate(layer_map, size=(h, w), mode='bilinear', align_corners=False)
                global_anomaly_map += layer_map_resized


            # kernel_size=9 e sigma=4.0 sono lo standard per MVTec a 224x224
            smoothed_map = TF.gaussian_blur(global_anomaly_map, kernel_size=[5,5], sigma=[1.0, 1.0])

            # mappa pulita dal rimore, prendo pixel peggiore
            flat_smoothed_map = smoothed_map.view(batch_size, -1)

            k = max(1, int(flat_smoothed_map.shape[1] * 0.01))
            topk_scores, _ = torch.topk(flat_smoothed_map, k, dim=1)
            
            # media di quest'area critica
            img_anomaly_scores = torch.mean(topk_scores, dim=1)
            # img_anomaly_scores, _ = torch.max(flat_smoothed_map, dim=1)

            """
            # top-K pooling
            flat_anomaly_map = global_anomaly_map.view(batch_size, -1)
            # 2% dell'immagine
            k = max(1, int(flat_anomaly_map.shape[1] * 0.02))
            # uso i k pixel con errore maggiore
            topk_scores, _ = torch.topk(flat_anomaly_map, k, dim=1)
            
            # lo score diventa la media dei pixel peggiori, in questo modo la dimensione della parte anomala non influisce sulla decisione
            img_anomaly_scores = torch.mean(topk_scores, dim=1)
            """
            # img_anomaly_scores = torch.mean(global_anomaly_map, dim=[1, 2, 3])
            # img_anomaly_scores, _ = torch.max(anomaly_map_resized.view(inputs.size(0), -1), dim=1)

            # se lo score supera la soglia è ANOMALA
            predictions = torch.where(img_anomaly_scores > THRESHOLD, 0, 1)

            # accuratezza
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            all_anomaly_scores.extend(img_anomaly_scores.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            print(f"Anomaly Scores: {img_anomaly_scores.tolist()}")
            print(f"Predizioni: {predictions.tolist()}")
            print(f"Target reali: {targets.tolist()}")
    

    accuracy = (correct_predictions / total_samples) * 100
    print(f"Test completato. Accuratezza Globale: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    targets_np = np.array(all_targets)
    scores_np = np.array(all_anomaly_scores)
    
    anomalia_come_classe_positiva = 1 - targets_np 
    
    auc_score = roc_auc_score(anomalia_come_classe_positiva, scores_np)
    print(f"ROC-AUC Score Globale: {auc_score * 100:.2f}%")

    # calcolo automatico della soglia ottimale (Youden)
    fpr, tpr, thresholds = roc_curve(anomalia_come_classe_positiva, scores_np)
    best_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[best_idx]
    print(f"Soglia ottimale calcolata: {optimal_threshold:.6f}")

    return accuracy, all_anomaly_scores, all_targets
