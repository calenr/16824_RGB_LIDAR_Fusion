import torch
import torch.nn as nn


class RgbLidarFusion(nn.Module):
    def __init__(self, args):
        super(RgbLidarFusion, self).__init__()

        res18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.image_enc = nn.Sequential(*list(res18.children())[:-2])
        self.image_enc.eval()
        for param in self.image_enc.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(1, 1)

    def forward(self, image: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """
        :param image: Batch x 3 x
        :param lidar: Batch x N x 3
        :return:
        """
        image_feat = self.image_enc(image)
        return image_feat
