import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_model(num_classes=7):
    # Carrega o modelo base pré-treinado
    model = retinanet_resnet50_fpn(weights="COCO_V1")
    
    # Número de canais e âncoras do modelo original
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    # Substitui o cabeçalho de classificação para suportar N classes
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model
