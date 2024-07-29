from torch import nn
import torch

class LocalizationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 5, stride=2),
            nn.GELU(),
            nn.Conv2d(8, 4, 5),
            nn.GELU()
        )

        # classification head: n_outputs = n_classes
        self.clf_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144,128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,10)
        )
        # box regression head: n_outputs = n_coords (4)
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144,128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,4)
        )

    def forward(self, pic: torch.Tensor):
        # backbone
        x = self.backbone(pic)
        # classification head
        clf_out = self.clf_head(x)
        # box regression head
        box_out = self.box_head(x)
        return clf_out, torch.sigmoid(box_out)
