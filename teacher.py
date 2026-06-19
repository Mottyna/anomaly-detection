from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from parameters import RETURN_NODES, THRESHOLD

class Teacher:
    def __init__(self, model):
        if model == "resnet":
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model = create_feature_extractor(base_model, return_nodes=RETURN_NODES)