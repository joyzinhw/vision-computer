import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_model(num_classes=7, weights="COCO_V1"):
    model = retinanet_resnet50_fpn(weights=weights)

    # tenta obter in_channels de forma robusta
    # preferência: usar cls_logits.in_channels (conv final do cabeçalho)
    try:
        in_channels = model.head.classification_head.cls_logits.in_channels
    except Exception:
        # fallback: inspeciona a primeira conv do bloco conv (pode estar encapsulado)
        try:
            first_conv = model.head.classification_head.conv[0]
            # se for Conv2dNormActivation (pytorch newer), pode ter .conv ou .op
            if hasattr(first_conv, "conv"):
                in_channels = first_conv.conv.in_channels
            elif hasattr(first_conv, "op"):
                in_channels = first_conv.op.in_channels
            else:
                # por fim tenta indexação direta
                in_channels = first_conv[0].in_channels
        except Exception as e:
            raise RuntimeError("Não foi possível inferir in_channels do classification_head.") from e

    # número de anchors (atributo presente no classification_head)
    num_anchors = model.head.classification_head.num_anchors

    # substitui o cabeçalho de classificação por um compatível com num_classes
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model
