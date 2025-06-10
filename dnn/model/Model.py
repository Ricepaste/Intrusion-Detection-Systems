from torch import Tensor, tensor
import torch.nn as nn

# 這是你指定作為 encoder 的部分
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.01)
        self.relu = nn.ReLU(inplace=True) # 通常全連接層後會接激活函數

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x) # 確保輸出是一個帶有激活的特徵向量
        return x

class SimSiam(nn.Module):
    def __init__(
        self,
        pretrained_model,  # 這裡會傳入 SimpleEncoder 實例
        encoder_output_dim=1024, # 這是 SimpleEncoder 的輸出維度
        projector_inner_dim=32,
    ):
        super(SimSiam, self).__init__()

        # create the encoder
        self.encoder = pretrained_model

        # build a 3-layer projector
        self.projector = nn.Sequential(
            # 注意：這裡不再有 Flatten()，因為 SimpleEncoder 輸出已經是二維 (batch_size, dim)
            nn.Linear(encoder_output_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(projector_inner_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(projector_inner_dim, projector_inner_dim, bias=False),
            nn.BatchNorm1d(projector_inner_dim, affine=False),  # third layer
        )  # output layer

        """
        according to the original paper,
        predictor's output and projector's output vector should be the same size to calculate loss.
        Meanwhile, the predictor's inner dimmension should be 1/4 of predictor's output dimmension.
        """
        predictor_output_dim = projector_inner_dim
        predictor_inner_dim = predictor_output_dim // 4
        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(predictor_output_dim, predictor_inner_dim, bias=False), # projector_output_dim 其實就是 projector_inner_dim
            nn.BatchNorm1d(predictor_inner_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(predictor_inner_dim, predictor_output_dim),
        )  # output layer

    def forward(self, x1, x2):
        # encoder 輸出已經是 (batch_size, encoder_output_dim)
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # SimSiam 損失函數的核心：餘弦相似度
        # loss = -(D(p1, z2).mean() + D(p2, z1).mean()) / 2
        # 其中 D(p, z) = -cosine_similarity(p, z)
        # z.detach() 實現停止梯度
        return p1, p2, z1.detach(), z2.detach()