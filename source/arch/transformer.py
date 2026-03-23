import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from function import normal, normal_style
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)  # 对于每个 token 的 512 维向量，nn.LayerNorm 计算其均值和方差，并进行归一化。
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        # 一个 1x1 卷积层，用于生成内容感知的位置编码。
        self.new_ps = nn.Conv2d(512, 512, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):
        # style, content 的原始形状假设都是 (B, C, H, W)
        B, C, H, W = style.shape

        # 1. content-aware positional embedding
        content_pool = self.averagepooling(content)  # (B, C, H', W')
        pos_c = self.new_ps(content_pool)  # (B, C, H', W')
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])  # (B, C, H, W)

        # 2. 展平特征: (B, C, H, W) -> (H*W, B, C)
        style_flat = style.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        content_flat = content.flatten(2).permute(2, 0, 1)

        pos_s_flat = None
        if pos_embed_s is not None:
            pos_s_flat = pos_embed_s.flatten(2).permute(2, 0, 1)

        pos_c_flat = None
        if pos_embed_c is not None:
            pos_c_flat = pos_embed_c.flatten(2).permute(2, 0, 1)

        # 3. 处理 mask：把 (B, H, W) 或 (B, 1, H, W) 展平到 (B, H*W)
        mask_flat = None
        if mask is not None:
            # 典型情况: mask 是 (B, 1, H, W) 或 (B, H, W)
            if mask.dim() == 4:
                # (B, 1, H, W) -> (B, H, W)
                if mask.size(1) == 1:
                    mask = mask[:, 0]
                else:
                    # 如果是 (B, C, H, W)，这里你要根据实际情况挑一个通道
                    mask = mask[:, 0]
            # 现在 mask 应该是 (B, H, W)
            mask_flat = mask.flatten(1)  # (B, H*W)

        # 4. 送入 encoder / decoder
        style_enc = self.encoder_s(style_flat, src_key_padding_mask=mask_flat, pos=pos_s_flat)
        content_enc = self.encoder_c(content_flat, src_key_padding_mask=mask_flat, pos=pos_c_flat)

        hs = self.decoder(
            content_enc, style_enc,
            memory_key_padding_mask=mask_flat,
            pos=pos_s_flat,
            query_pos=pos_c_flat
        )[0]  # (HW, B, C)

        # 5. (HW, B, C) -> (B, C, H, W)，用原始的 H, W，不用 sqrt 猜
        hs = hs.permute(1, 2, 0).contiguous()  # (B, C, HW)
        hs = hs.view(B, C, H, W)  # (B, C, H, W)

        return hs


# 实现 Transformer 编码器的前向传播，依次通过 num_layers 个编码器层处理输入序列
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # 相当于transformerencodelayer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


# TransformerDecoder由多个解码器层（TransformerDecoderLayer）堆叠而成。
class TransformerDecoder(nn.Module):  # 结合编码器输入，进行多层交叉注意力

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


# 负责对输入序列（例如展平后的特征图）进行Self-Attention、Feed-Forward Network, FFN）处理。
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    # 将位置编码（pos）添加到输入张量（tensor）上
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # Post-LN（默认）：归一化在残差连接之后，传统 Transformer 使用的模式。
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 将位置编码加到输入上，q 和 k 相同 下列结构和文中提到的transformer Encoder一样
        q = k = self.with_pos_embed(src, pos)
        # q = k = src
        # print(q.size(),k.size(),src.size())
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    # Pre-LN：归一化在残差连接之前，近年研究表明 Pre-LN 在深层网络中更稳定。
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# 将内容特征（作为查询，query）和风格特征（作为记忆，memory）融合
class TransformerDecoderLayer(nn.Module):
    # d_model：模型维度、nhead：多头注意力的头数、dim_feedforward=2048：FFN 的隐藏层维度

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # d_model embedding dim
        # 自注意力模块，用于目标序列（tgt）的自注意力。
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 交叉注意力模块，用于融合目标序列（tgt）和记忆序列（memory）
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # tgt内容特征），(L, B, D)（L=1024, B 是批量大小，D=512）、memory：记忆序列（风格特征）
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  # FNN
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


"""
创建 N 个给定模块（module）的深拷贝（deep copy），并将它们封装在 nn.ModuleList 中
为什么需要独立层？
    每个编码器层（或解码器层）需要有独立的参数，以便学习不同的特征。
    如果所有层共享参数，模型的表达能力会受到限制。
"""


def _get_clones(module, N):  # 克隆 N 份模块
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):  # 根据 args 参数构建并返回一个完整的 Transformer 编码-解码网络结构
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):  # 激活函数映射器

    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
