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


class ResidualBlock(nn.Module):
    """
    blocco residuo classico per mantenere ed elaborare le feature 
    nel collo di bottiglia senza perdere dettagli spaziali.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self.target_channels = 512 
        
        self.proj1 = nn.Conv2d(256, self.target_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(512, self.target_channels, kernel_size=1) 
        self.proj3 = nn.Conv2d(1024, self.target_channels, kernel_size=1)
        
        self.reduction = nn.Sequential(
            nn.Conv2d(self.target_channels * 3, self.target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(self.target_channels),
            ResidualBlock(self.target_channels)
        )

    def forward(self, feats):
        f1, f2, f3 = feats['feat1'], feats['feat2'], feats['feat3']
        
        # allineamento spaziale alla risoluzione minima
        target_size = f3.shape[2:] 
        
        m1 = F.interpolate(self.proj1(f1), size=target_size, mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.proj2(f2), size=target_size, mode='bilinear', align_corners=False)
        m3 = self.proj3(f3)
        
        mff = torch.cat([m1, m2, m3], dim=1)
        
        x = self.reduction(mff)
        oce = self.res_blocks(x)
        return oce


class ReverseDistillationStudent(nn.Module):
    """
    decoder a espansione progressiva: 14x14 -> 28x28 -> 56x56.
    l'interpolazione avviene prima della convoluzione.
    """
    def __init__(self):
        super().__init__()
        self.bottleneck = Bottleneck()
        
        # decoder per feat3
        self.decoder_feat3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # decoder per feat2
        self.decoder_feat2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # decoder per feat1
        self.decoder_feat1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, teacher_outputs):
        # OCE parte dalla risoluzione minima (14x14)
        oce = self.bottleneck(teacher_outputs)
        
        # ricostruzione feat3
        s_feat3 = self.decoder_feat3(oce)
        
        # ricostruzione feat2 con upsampling
        up2 = F.interpolate(s_feat3, size=teacher_outputs['feat2'].shape[2:], mode='bilinear', align_corners=False)
        s_feat2 = self.decoder_feat2(up2)
        
        # ricostruzione feat1 con upsampling
        up1 = F.interpolate(s_feat2, size=teacher_outputs['feat1'].shape[2:], mode='bilinear', align_corners=False)
        s_feat1 = self.decoder_feat1(up1)
        
        return {
            'feat1': s_feat1,
            'feat2': s_feat2,
            'feat3': s_feat3
        }