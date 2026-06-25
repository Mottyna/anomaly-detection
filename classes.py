from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import torch.nn.functional as F
import torch
from parameters import RETURN_NODES

class Teacher:
    def __init__(self, model):
        if model == "resnet":
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model = create_feature_extractor(base_model, return_nodes=RETURN_NODES)


class ProjectorWrapper(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = extractor
                self.projectors = nn.ModuleDict({
                    'feat1': nn.Conv2d(64, 256, kernel_size=1),
                    'feat2': nn.Conv2d(128, 512, kernel_size=1),
                    'feat3': nn.Conv2d(256, 1024, kernel_size=1)
                })

            def forward(self, x):
                feats = self.extractor(x)
                return {k: self.projectors[k](v) for k, v in feats.items()}


class Bottleneck(nn.Module):
    """
    MFF: prende feat1, feat2, feat3, le unisce spazialmente.
    OCE: proietta il risultato in un One-Class Embedding.
    """
    def __init__(self):
        super().__init__()
        # ResNet50: feat1=256 canali, feat2=512 canali, feat3=1024 canali
        self.target_channels = 512 
        
        # opzionale
        self.proj1 = nn.Conv2d(256, self.target_channels, kernel_size=1)
        self.proj2 = nn.Identity() # Ha 512 canali
        self.proj3 = nn.Conv2d(1024, self.target_channels, kernel_size=1)
        
        # dopo la concatenazione 512 * 3 = 1536 canali
        # il blocco OCE riduce dimensionalità
        self.oce = nn.Sequential(
            nn.Conv2d(self.target_channels * 3, self.target_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        f1, f2, f3 = feats['feat1'], feats['feat2'], feats['feat3']
        target_size = f2.shape[2:] # usiamo la dimensione del blocco centrale
        
        # Multi-scale Feature Fusion (MFF) con allineamento spaziale
        m1 = F.interpolate(self.proj1(f1), size=target_size, mode='bilinear', align_corners=False)
        m2 = self.proj2(f2)
        m3 = F.interpolate(self.proj3(f3), size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenazione lungo i canali
        mff = torch.cat([m1, m2, m3], dim=1)
        
        # Passaggio nel collo di bottiglia (OCE)
        return self.oce(mff)


class ReverseDistillationStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = Bottleneck()
        
        # decoder 3: prende l'OCE (512 canali, 28x28) e genera feat3 (1024 canali, 14x14)
        self.d3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # decoder 2: prende l'output di d3 (1024 canali) e genera feat2 (512 canali, 28x28)
        self.d2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # decoder 1: prende l'output di d2 (512 canali) e genera feat1 (256 canali, 56x56)
        self.d1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, teacher_outputs):
        # MFF + OCE
        oce = self.bottleneck(teacher_outputs)
        
        # FLUSSO REVERSE SEQUENZIALE
        out_d3 = self.d3(oce)
        s_feat3 = F.interpolate(out_d3, size=teacher_outputs['feat3'].shape[2:], mode='bilinear', align_corners=False)

        out_d2 = self.d2(s_feat3) 
        s_feat2 = F.interpolate(out_d2, size=teacher_outputs['feat2'].shape[2:], mode='bilinear', align_corners=False)

        out_d1 = self.d1(s_feat2)
        s_feat1 = F.interpolate(out_d1, size=teacher_outputs['feat1'].shape[2:], mode='bilinear', align_corners=False)
        
        return {
            'feat1': s_feat1,
            'feat2': s_feat2,
            'feat3': s_feat3
        }