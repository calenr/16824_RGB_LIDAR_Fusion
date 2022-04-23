import torch
import torch.nn as nn


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()

        res18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.image_enc = nn.Sequential(*list(res18.children())[:-3])
        # TODO: should we finetune the img encoder?
        # self.image_enc.eval()
        # for param in self.image_enc.parameters():
        #     param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        :param image: Batch x 3 x
        :return: image_feat
        """
        image_feat = self.image_enc(image)
        return image_feat
