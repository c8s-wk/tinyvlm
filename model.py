import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class VLM(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # 图像编码器 (ResNet18, 去掉分类头)
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.img_fc = nn.Linear(resnet.fc.in_features, embed_dim)

        # 文本编码器 (BERT-base)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.txt_fc = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def forward(self, images, input_ids, attention_mask):
        # Image branch
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)  # (B, 512)
        img_embed = self.img_fc(img_feat)                              # (B, embed_dim)

        # Text branch
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.pooler_output                               # CLS向量
        txt_embed = self.txt_fc(txt_feat)                              # (B, embed_dim)

        # L2 normalize
        img_embed = nn.functional.normalize(img_embed, p=2, dim=-1)
        txt_embed = nn.functional.normalize(txt_embed, p=2, dim=-1)

        return img_embed, txt_embed
