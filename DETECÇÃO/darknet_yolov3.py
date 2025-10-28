import math
import torch
import torch.nn as nn
from build_utils.layers import FeatureConcat, WeightedFeatureFusion
from build_utils.parse_config import parse_model_cfg

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    modules_defs.pop(0)  # remove os hiperparâmetros do cfg

    output_filters = [3]  # canais iniciais da imagem RGB
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()
        filters = output_filters[-1]  # padrão: manter os filtros da camada anterior

        # -------------------------------------------------
        # [convolutional]
        # -------------------------------------------------
        if mdef["type"] == "convolutional":
            bn = mdef.get("batch_normalize", 0)
            filters = mdef["filters"]
            k = mdef["size"]
            stride = mdef["stride"]
            pad = (k - 1) // 2 if mdef.get("pad", 0) else 0

            in_channels = output_filters[-1]
            if in_channels <= 0:
                raise ValueError(f"Camada {i}: Número de canais de entrada inválido: {in_channels}")

            modules.add_module(
                "Conv2d",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=k,
                    stride=stride,
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            if mdef.get("activation") == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # -------------------------------------------------
        # [upsample]
        # -------------------------------------------------
        elif mdef["type"] == "upsample":
            modules = nn.Upsample(scale_factor=mdef["stride"], mode="nearest")
            filters = output_filters[-1]

        # -------------------------------------------------
        # [route] (corrigido e robusto)
        # -------------------------------------------------
        elif mdef["type"] == "route":
            layers_str = str(mdef["layers"]).replace(" ", "")
            layers = []
            for x in layers_str.split(","):
                if x.strip() != "":
                    try:
                        # converte '-4.0' para -4
                        layers.append(int(float(x)))
                    except ValueError:
                        raise ValueError(f"Erro ao converter índice '{x}' na camada {i} (route). Corrija o arquivo .cfg.")
            layers = [l if l >= 0 else i + l for l in layers]

            if len(layers) == 1:
                filters = output_filters[layers[0]]
                print(f"Camada route {i}: conectando camada {layers[0]} -> {filters} filtros")
            else:
                filters = sum([output_filters[l] for l in layers])
                print(f"Camada route {i}: concatenando camadas {layers} -> {filters} filtros")

            routs.extend([l for l in layers if 0 <= l < len(output_filters)])
            modules = nn.Identity()
            modules.layers = layers  # salva os índices das camadas

        # -------------------------------------------------
        # [shortcut] (corrigido)
        # -------------------------------------------------
        elif mdef["type"] == "shortcut":
            try:
                from_layer = int(float(mdef["from"]))  # converte '-4.0' -> -4
            except ValueError:
                raise ValueError(f"Erro ao converter 'from={mdef['from']}' na camada {i} (shortcut). Corrija o arquivo .cfg.")

            from_layer = from_layer if from_layer >= 0 else i + from_layer

            if from_layer < 0 or from_layer >= len(output_filters):
                raise ValueError(f"Camada {i}: Índice 'from' inválido {from_layer}")

            filters = output_filters[-1]
            routs.append(from_layer)
            modules = WeightedFeatureFusion(layers=[from_layer], weight="weights_type" in mdef)
            print(f"Camada shortcut {i}: conectando com camada {from_layer}")

        # -------------------------------------------------
        # [yolo]
        # -------------------------------------------------
        elif mdef["type"] == "yolo":
            yolo_index += 1
            stride = [32, 16, 8]

            mask = mdef["mask"]
            if isinstance(mask, str):
                mask = [int(x) for x in mask.split(",")]

            anchors = mdef["anchors"]
            if isinstance(anchors, str):
                anchors = [float(a) for a in anchors.replace(" ", "").split(",")]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

            modules = YOLOLayer(
                anchors=anchors,
                nc=mdef["classes"],
                img_size=img_size,
                stride=stride[yolo_index],
            )
            filters = output_filters[-1]
            print(f"Camada YOLO {i}: {len(anchors)} anchors, {mdef['classes']} classes")

        else:
            print(f"Aviso: Tipo de camada desconhecido '{mdef['type']}' na camada {i}")
            filters = output_filters[-1]

        module_list.append(modules)
        output_filters.append(filters)
        print(f"Camada {i:3d}: {mdef['type']:12s} -> {filters:4d} filtros")

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        if i < len(routs_binary):
            routs_binary[i] = True

    return module_list, routs_binary


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, stride):
        super().__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx, self.ny = 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def forward(self, p):
        bs, nc, ny, nx = p.shape

        expected_channels = self.na * self.no
        if nc != expected_channels:
            raise ValueError(f"YOLOLayer: Esperado {expected_channels} canais, mas recebeu {nc} canais. "
                             f"na={self.na}, no={self.no}, nc={self.nc}")

        if (self.nx, self.ny) != (nx, ny) or self.grid is None:
            self.create_grids((nx, ny), p.device)

        p = p.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p

        io = p.clone()
        io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
        io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
        io[..., :4] *= self.stride
        torch.sigmoid_(io[..., 4:])
        return io.view(bs, -1, self.no), p

    def create_grids(self, ng=(13, 13), device="cpu"):
        self.nx, self.ny = ng
        yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                 torch.arange(self.nx, device=device)], indexing='ij')
        self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        self.anchor_vec = self.anchor_vec.to(device)
        self.anchor_wh = self.anchor_wh.to(device)


class Darknet(nn.Module):
    def __init__(self, cfg, img_size=(416, 416)):
        super().__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']

    def forward(self, x):
        yolo_out, out = [], []

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__

            if isinstance(module, nn.Identity):
                layers = getattr(module, "layers", [])
                feats = [out[l] for l in layers if l < len(out) and out[l] is not None]
                x = torch.cat(feats, 1) if len(feats) > 1 else feats[0]

            elif name == 'WeightedFeatureFusion':
                layers = module.layers
                for layer_idx in layers:
                    if layer_idx < len(out) and out[layer_idx] is not None:
                        if module.weight:
                            x = x * module.w[0] + out[layer_idx] * module.w[1]
                        else:
                            x = x + out[layer_idx]

            elif name == 'YOLOLayer':
                yolo_out.append(module(x))
                out.append(None)
                continue

            else:
                x = module(x)

            out.append(x if i < len(self.routs) and self.routs[i] else None)

        if self.training:
            return yolo_out

        x, p = zip(*yolo_out)
        return torch.cat(x, 1), p
