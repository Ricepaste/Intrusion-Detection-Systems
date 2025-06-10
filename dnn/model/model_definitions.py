import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class SimSiam(nn.Module):
    def __init__(
        self,
        pretrained_model,
        encoder_output_dim=1024,
        projector_inner_dim=32,
    ):
        super(SimSiam, self).__init__()

        self.encoder = pretrained_model

        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_inner_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_inner_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim, affine=False),
        )

        predictor_output_dim = projector_inner_dim
        predictor_inner_dim = predictor_output_dim // 4
        self.predictor = nn.Sequential(
            nn.Linear(predictor_output_dim, predictor_inner_dim, bias=False),
            nn.BatchNorm1d(predictor_inner_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_inner_dim, predictor_output_dim),
        )

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

# SimSiam 損失函數
def simsiam_loss(p1, p2, z1, z2):
    loss_p1_z2 = -F.cosine_similarity(p1, z2, dim=-1).mean()
    loss_p2_z1 = -F.cosine_similarity(p2, z1, dim=-1).mean()
    return (loss_p1_z2 + loss_p2_z1) / 2

# 下游分類器
class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, encoder_output_dim, num_classes):
        super(DownstreamClassifier, self).__init__()
        self.encoder = encoder
        # 凍結 encoder 參數 (通常預訓練後不會再訓練 encoder)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.classifier = nn.Linear(encoder_output_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output