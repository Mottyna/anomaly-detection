from torchvision.models import (resnet50, ResNet50_Weights,
                                wide_resnet50_2, Wide_ResNet50_2_Weights,
                                resnet34, ResNet34_Weights,
                                resnet18, ResNet18_Weights)
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import torch.nn.functional as F
import torch
from parameters import RETURN_NODES


class Teacher:
    def __init__(self, model_name):
        self.model_name = model_name
        self.return_nodes = RETURN_NODES
        # ResNet50
        if model_name == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.channels = {'feat1': 256, 'feat2': 512, 'feat3': 1024}
        # WideResNet50
        elif model_name == "wideresnet50":
            base_model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
            self.channels = {'feat1': 256, 'feat2': 512, 'feat3': 1024}
        # ResNet18
        elif model_name == "resnet18":
            base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.channels = {'feat1': 64, 'feat2': 128, 'feat3': 256}

        self.model = create_feature_extractor(base_model, return_nodes=self.return_nodes)


class ProjectorWrapper(nn.Module):
    def __init__(self, extractor, student_channels, teacher_channels):
        """
        proietta le feature dello student per allinearle ai canali del teacher.
        """
        super().__init__()
        self.extractor = extractor
        self.projectors = nn.ModuleDict({
            k: nn.Conv2d(student_channels[k], teacher_channels[k], kernel_size=1)
            for k in teacher_channels.keys()
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
    def __init__(self, teacher_channels):
        super().__init__()
        self.target_channels = 512 
        
        self.proj1 = nn.Conv2d(teacher_channels['feat1'], self.target_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(teacher_channels['feat2'], self.target_channels, kernel_size=1) 
        self.proj3 = nn.Conv2d(teacher_channels['feat3'], self.target_channels, kernel_size=1)
        
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
    def __init__(self, teacher_channels):
        super().__init__()
        t_ch1 = teacher_channels['feat1']
        t_ch2 = teacher_channels['feat2']
        t_ch3 = teacher_channels['feat3']
        
        bottleneck_out = 512 
        self.bottleneck = Bottleneck(teacher_channels=teacher_channels) 

        # SK-RD4AD non-corresponding skips
        # feat3 decoder riceve solo l'output del bottleneck
        self.decoder_feat3 = self._make_decoder_block(bottleneck_out, t_ch3)
        # feat2 decoder riceve s_feat3 + teacher feat3
        self.decoder_feat2 = self._make_decoder_block(t_ch3 * 2, t_ch2)
        # 3. feat1 decoder riceve s_feat2 + teacher feat2
        self.decoder_feat1 = self._make_decoder_block(t_ch2 * 2, t_ch1)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, teacher_outputs):
        oce = self.bottleneck(teacher_outputs)
        
        # decode feat3 (no skips)
        s_feat3 = self.decoder_feat3(oce)
        
        # decode feat2 + skip (-> feat3 del teacher)
        concat_feat3 = torch.cat([s_feat3, teacher_outputs['feat3']], dim=1)
        up3 = F.interpolate(concat_feat3, size=teacher_outputs['feat2'].shape[2:], mode='bilinear', align_corners=False)
        s_feat2 = self.decoder_feat2(up3)
        
        # decode feat1 + skip (-> feat2 del teacher)
        concat_feat2 = torch.cat([s_feat2, teacher_outputs['feat2']], dim=1)
        up1 = F.interpolate(concat_feat2, size=teacher_outputs['feat1'].shape[2:], mode='bilinear', align_corners=False)
        s_feat1 = self.decoder_feat1(up1)
        
        return { 'feat1': s_feat1, 
                'feat2': s_feat2, 
                'feat3': s_feat3 }