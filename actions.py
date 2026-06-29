import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from parameters import RETURN_NODES, MEAN, STD, WEIGHTS
import matplotlib.pyplot as plt
import os

def train(teacher, student, train_loader, epochs, learning_rate, T, device, reverse_distillation=False):

    teacher.eval()
    student.train()

    teacher = teacher.to(device)
    student = student.to(device)

    # criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=1e-5)
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
            if not reverse_distillation:
                student_outputs = student(inputs)
            else:
                student_outputs = student(teacher_outputs)

            # calcolo della loss tra le mappe di feature nei diversi layers
            loss = 0.0
            for key in RETURN_NODES.values():
                t_feat = teacher_outputs[key]
                s_feat = student_outputs[key]

                # normalizzato
                t_feat_norm = F.normalize(t_feat, p=2, dim=1)
                s_feat_norm = F.normalize(s_feat, p=2, dim=1)

                # COSINE SIMILARITY
                cos_sim = F.cosine_similarity(s_feat_norm, t_feat_norm, dim=1)
                if not reverse_distillation:
                    loss += (1 - cos_sim).mean()
                else:
                    loss += ((1 - cos_sim)*WEIGHTS[key]).mean()

                #loss += criterion(s_feat_norm, t_feat_norm)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()    # aggiorno il learning rate
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    return student


def test(teacher, student, test_loader, device, threshold, reverse_distillation=False):
    """
    """
    teacher.eval()
    student.eval()

    total_samples = 0
    correct_predictions = 0
    
    all_anomaly_scores = []
    all_targets = []

    all_pixel_scores = []
    all_pixel_targets = []

    print("\n--- Avvio fase di test ---")
    with torch.no_grad():
        for batch_idx, (inputs, masks, targets) in enumerate(test_loader):
            targets = targets.to(device)
            masks = masks.to(device)
            inputs = inputs.to(device)
            
            t_outs = teacher(inputs)

            if not reverse_distillation:
                s_outs = student(inputs)
            else:
                s_outs = student(t_outs)
            
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
                if not reverse_distillation:
                    global_anomaly_map += layer_map_resized
                else:
                    # somma pesate dei layers
                    global_anomaly_map += layer_map_resized*WEIGHTS[key]


            # kernel_size=9 e sigma=4.0 sono lo standard per MVTec a 224x224
            smoothed_map = TF.gaussian_blur(global_anomaly_map, kernel_size=[5,5], sigma=[1.0, 1.0])
            # smoothed_map = TF.gaussian_blur(global_anomaly_map, kernel_size=[15,15], sigma=[4.0, 4.0])

            # per evitare l'effetto bordo anomalo
            margin = 10
            smoothed_map[:, :, :margin, :] = 0.0  # bordo superiore
            smoothed_map[:, :, -margin:, :] = 0.0 # bordo inferiore
            smoothed_map[:, :, :, :margin] = 0.0  # bordo sinistro
            smoothed_map[:, :, :, -margin:] = 0.0 # bordo destro

            # mappa pulita dal rumore
            flat_smoothed_map = smoothed_map.view(batch_size, -1)

            # era 0.01 prima
            k = max(1, int(flat_smoothed_map.shape[1] * 0.005))
            topk_scores, _ = torch.topk(flat_smoothed_map, k, dim=1)
            
            # media di quest'area critica
            img_anomaly_scores = torch.mean(topk_scores, dim=1)

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
            predictions = torch.where(img_anomaly_scores > threshold, 0, 1)

            save_anomaly_visualizations(images=inputs,
                                        masks=masks,
                                        anomaly_maps=smoothed_map, # smoothed_map per una visualizzazione pulita senza rumore
                                        targets=targets, 
                                        predictions=predictions, 
                                        batch_idx=batch_idx,
                                        save_dir="risultati")

            # accuratezza
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            all_anomaly_scores.extend(img_anomaly_scores.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            all_pixel_scores.extend(smoothed_map.cpu().numpy().flatten())
            all_pixel_targets.extend((masks > 0.5).cpu().numpy().astype(np.int32).flatten())

    accuracy = (correct_predictions / total_samples) * 100
    print(f"[TEST] Test completato. Accuratezza globale: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    targets_np = np.array(all_targets)
    scores_np = np.array(all_anomaly_scores)
    pixel_targets_np = np.array(all_pixel_targets)
    pixel_scores_np = np.array(all_pixel_scores)
    
    anomalia_come_classe_positiva = 1 - targets_np 
    
    auc_score = roc_auc_score(anomalia_come_classe_positiva, scores_np)
    print(f"[TEST] ROC-AUC score globale (image level): {auc_score * 100:.2f}%")

    pixel_auc_score = roc_auc_score(pixel_targets_np, pixel_scores_np)
    print(f"[TEST] ROC-AUC score locale (pixel level): {pixel_auc_score * 100:.2f}%")

    # calcolo automatico della soglia ottimale (Youden)
    fpr, tpr, thresholds = roc_curve(anomalia_come_classe_positiva, scores_np)
    best_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[best_idx]
    print(f"[TEST] Threshold ottimale calcolato nella fase di test (tramite AUC-ROC): {optimal_threshold:.6f}")

    # PR-AUC 
    precision_curve, recall_curve, _ = precision_recall_curve(anomalia_come_classe_positiva, scores_np)
    pr_auc = auc(recall_curve, precision_curve)

    # PRECISION, RECALL E F1-SCORE
    binary_predictions = (scores_np >= threshold).astype(int)

    precision = precision_score(anomalia_come_classe_positiva, binary_predictions, zero_division=0)
    recall = recall_score(anomalia_come_classe_positiva, binary_predictions, zero_division=0)
    f1 = f1_score(anomalia_come_classe_positiva, binary_predictions, zero_division=0)

    print(f"[TEST] PR-AUC score globale: {pr_auc * 100:.2f}%")
    print(f"[TEST] Precision: {precision * 100:.2f}% (Se dico 'Difetto', quante volte e' corretto?)")
    print(f"[TEST] Recall: {recall * 100:.2f}% (Di tutti i difetti reali, quanti ne ho trovati?)")
    print(f"[TEST] F1-Score: {f1 * 100:.2f}% (Media armonica tra Precision e Recall)")

    return accuracy, all_anomaly_scores, all_targets


def save_anomaly_visualizations(images, masks, anomaly_maps, targets, predictions, batch_idx, save_dir="results"):
    """
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size = images.shape[0]
    
    for i in range(batch_size):
        # ripristina l'immagine originale
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * STD) + MEAN
        img = np.clip(img, 0, 1)  # forza i valori nel range [0, 1]
        
        gt_mask = masks[i].cpu().numpy().squeeze()

        # prepara l'anomaly map
        amap = anomaly_maps[i].cpu().numpy().squeeze() # rimuove la dimensione del canale (1, H, W) -> (H, W)
        
        # normalizzazione locale della mappa per renderla più visibile (range 0-1)
        if amap.max() - amap.min() > 0:
            amap_normalized = (amap - amap.min()) / (amap.max() - amap.min())
        else:
            amap_normalized = amap

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # pannello 1: originale
        axes[0].imshow(img)
        axes[0].set_title(f"Originale (Target: {targets[i].item()})")
        axes[0].axis('off')

        # pannello 2: ground truth
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth (Maschera Reale)")
        axes[1].axis('off')
        
        # pannello 2: Anomaly Map
        axes[2].imshow(amap_normalized, cmap='jet')
        axes[2].set_title(f"Anomaly Map (Heatmap), predizione: {'Normale' if predictions[i].item() == 1 else 'Anomalo'}")
        axes[2].axis('off')

        img_id = batch_idx * batch_size + i
        plt.savefig(f"{save_dir}/anomaly_sample_{img_id}.png", bbox_inches='tight', dpi=150)
        plt.close() # chiude la figura per liberare memoria RAM


def generate_synthetic_anomalies(inputs):
    """
    genera difetti sintetici nelle immagini di dimensioni e tipo variabile.
    """
    batch_size, c, h, w = inputs.shape
    targets = torch.ones(batch_size, dtype=torch.long, device=inputs.device) # 1 = Good
    corrupted_inputs = inputs.clone()
    
    num_anomalies = batch_size // 2
    for i in range(num_anomalies):
        targets[i] = 0 # 0 = anomalo
        
        # dimensioni del difetto variabili (dal 2% al 15% dell'immagine)
        patch_h = torch.randint(int(h * 0.02), int(h * 0.15), (1,)).item()
        patch_w = torch.randint(int(w * 0.02), int(w * 0.15), (1,)).item()
        
        # posizione
        top = torch.randint(0, h - patch_h, (1,)).item()
        left = torch.randint(0, w - patch_w, (1,)).item()
        
        scelta_difetto = torch.rand(1).item()

        """
        if torch.rand(1).item() > 0.5:
            # macchia di colore
            random_color = torch.rand(c, 1, 1, device=inputs.device) * 4.0 - 2.0 
            corrupted_inputs[i, :, top:top+patch_h, left:left+patch_w] = random_color
        else:
            # rumore
            noise = torch.randn(c, patch_h, patch_w, device=inputs.device) * 0.8
            corrupted_inputs[i, :, top:top+patch_h, left:left+patch_w] += noise
        """

        if scelta_difetto > 0.5:
            # CUT-PASTE (copia una patch da un'altra parte e la incolla qui)
            # simula graffi strutturali / difetti di trama
            top_src = torch.randint(0, h - patch_h, (1,)).item()
            left_src = torch.randint(0, w - patch_w, (1,)).item()
            
            patch = inputs[i, :, top_src:top_src+patch_h, left_src:left_src+patch_w]
            corrupted_inputs[i, :, top:top+patch_h, left:left+patch_w] = patch
            
        else:
            # ALPHA BLENDING NOISE / COLOR (macchie sfocate / aloni)
            # trasparenza casuale tra il 10% e il 35%
            alpha = torch.rand(1).item() * 0.25 + 0.10 
            
            if torch.rand(1).item() > 0.5:
                # alone di colore
                mod = torch.rand(c, 1, 1, device=inputs.device) * 2.0 - 1.0
            else:
                # rumore
                mod = torch.randn(c, patch_h, patch_w, device=inputs.device) * 0.5
                
            original_patch = corrupted_inputs[i, :, top:top+patch_h, left:left+patch_w]
            corrupted_inputs[i, :, top:top+patch_h, left:left+patch_w] = (1 - alpha) * original_patch + alpha * mod
            
    return corrupted_inputs, targets


def get_validation_threshold(teacher, student, val_loader, device, reverse_distillation=False):
    """
    inserisce anomalie sintetiche nei dati di validazione per trovare la 
    soglia che massimizza la separazione geometrica (Youden).
    """
    teacher.eval()
    student.eval()
    
    all_anomaly_scores = []
    all_targets = []
    
    print("\n--- Calibrazione del threshold ---")
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            
            # genera anomalie artificiali
            inputs_corrupted, targets = generate_synthetic_anomalies(inputs)
            
            t_outs = teacher(inputs_corrupted)
            if not reverse_distillation:
                s_outs = student(inputs_corrupted)
            else:
                s_outs = student(t_outs)
                
            batch_size, _, h, w = inputs_corrupted.shape
            global_anomaly_map = torch.zeros((batch_size, 1, h, w), device=device)
            
            # anomaly map
            for key in RETURN_NODES.values():
                t_feat = F.normalize(t_outs[key], p=2, dim=1)
                s_feat = F.normalize(s_outs[key], p=2, dim=1)
                cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1).unsqueeze(1)
                layer_map = 1 - cos_sim
                layer_map_resized = F.interpolate(layer_map, size=(h, w), mode='bilinear', align_corners=False)
                
                if not reverse_distillation:
                    global_anomaly_map += layer_map_resized
                else:
                    global_anomaly_map += layer_map_resized * WEIGHTS[key]
                    
            smoothed_map = TF.gaussian_blur(global_anomaly_map, kernel_size=[5,5], sigma=[1.0, 1.0])
            # per evitare l'effetto bordo anomalo
            margin = 10
            smoothed_map[:, :, :margin, :] = 0.0  # bordo superiore
            smoothed_map[:, :, -margin:, :] = 0.0 # bordo inferiore
            smoothed_map[:, :, :, :margin] = 0.0  # bordo sinistro
            smoothed_map[:, :, :, -margin:] = 0.0 # bordo destro
            flat_smoothed_map = smoothed_map.view(batch_size, -1)
            k = max(1, int(flat_smoothed_map.shape[1] * 0.005))
            topk_scores, _ = torch.topk(flat_smoothed_map, k, dim=1)
            img_anomaly_scores = torch.mean(topk_scores, dim=1)
            
            all_anomaly_scores.extend(img_anomaly_scores.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    scores_np = np.array(all_anomaly_scores)
    targets_np = np.array(all_targets)
    
    # 1 = anomalia nella metrica ROC
    anomalia_come_classe_positiva = 1 - targets_np 
    
    fpr, tpr, thresholds = roc_curve(anomalia_come_classe_positiva, scores_np)
    best_idx = np.argmax(tpr - fpr)
    threshold = thresholds[best_idx]
    
    print(f"[VALIDAZIONE] Threshold ottimale: {threshold:.6f}")
    return threshold
