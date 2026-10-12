from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class P4ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        proj_dim = 256
        self.fc5 = nn.Linear(512, proj_dim)
        self.fc4 = nn.Linear(256, proj_dim)
        self.fc3 = nn.Linear(128, proj_dim)

        self.scale_attn = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=1),
        )

        self.cls = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        f3 = self.layer2(x)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)

        logits5 = torch.flatten(F.adaptive_avg_pool2d(f5, 1), 1)
        logits5 = self.fc5(logits5)

        logits4 = torch.flatten(F.adaptive_avg_pool2d(f4, 1), 1)
        logits4 = self.fc4(logits4)

        logits3 = torch.flatten(F.adaptive_avg_pool2d(f3, 1), 1)
        logits3 = self.fc3(logits3)

        
        cat = torch.cat([logits3, logits4, logits5], dim=1)
        w = self.scale_attn(cat)

        fused = (
            w[:, 0:1] * logits3 +
            w[:, 1:2] * logits4 +
            w[:, 2:3] * logits5
        )

        logits = self.cls(fused)
        return logits, w


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


device = "cpu"

model = P4ResNetClassifier(num_classes=1)
state = torch.load(Path(__file__).resolve().parent / "yeni.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# Use this if you only want classifier output.
export_model = LogitsOnly(model)
export_model.eval()

dummy = torch.randn(1, 3, 160, 160)

torch.onnx.export(
    export_model,
    dummy,
    str(Path(__file__).resolve().parent / "classifier.onnx"),
    input_names=["images"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
    dynamo=False,
)

print(f"Saved {Path(__file__).resolve().parent / 'classifier.onnx'}")
