from torchvision.models import resnet50, ResNet50_Weights

class Teacher:
    def __init__(self, model):
        if model == "resnet":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)