import math
import torch
import torch.nn as nn
from build_utils.layers import FeatureConcat, WeightedFeatureFusion

from build_utils.parse_config import parse_model_cfg

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    modules_defs.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()
        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            k = mdef["size"]
            stride = mdef["stride"]
            modules.add_module("Conv2d", nn.Conv2d(
                in_channels=output_filters[-1],
                out_channels=filters,
                kernel_size=k,
                stride=stride,
                padding=k // 2 if mdef["pad"] else 0,
                bias=not bn))
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
        elif mdef["type"] == "maxpool":
            modules = nn.MaxPool2d(kernel_size=mdef["size"], stride=mdef["stride"], padding=(mdef["size"] - 1) // 2)
        elif mdef["type"] == "upsample":
            modules = nn.Upsample(scale_factor=mdef["stride"])
        elif mdef["type"] == "route":
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)
        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)
        elif mdef["type"] == "yolo":
            yolo_index += 1
            stride = [32, 16, 8]
            modules = YOLOLayer(mdef["anchors"][mdef["mask"]], mdef["classes"], img_size, stride[yolo_index])
        module_list.append(modules)
        output_filters.append(filters)
    routs_binary = [False] * len(modules_defs)
    for i in routs:
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
        bs, _, ny, nx = p.shape
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
                                 torch.arange(self.nx, device=device)])
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
            if module.__class__.__name__ in ["WeightedFeatureFusion", "FeatureConcat"]:
                x = module(x, out)
            elif module.__class__.__name__ == "YOLOLayer":
                yolo_out.append(module(x))
            else:
                x = module(x)
            out.append(x if self.routs[i] else [])
        if self.training:
            return yolo_out
        x, p = zip(*yolo_out)
        return torch.cat(x, 1), p
