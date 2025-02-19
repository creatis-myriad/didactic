# Disable flak8 entirely because this file was mostly copied over from external sources and
# the quality of the codebase and documentation is not up to par with the project's standards
# flake8: noqa

# This code is copied (and its configuration adapted) from the official IRENE implementation: https://github.com/RL4M/IRENE
# Only the encoder architecture was needed, so copying the code allowed us to avoid dealing with unnecessary
# dependencies brought by other parts of the IRENE codebase.

import copy
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import Dropout, LayerNorm, Linear, Softmax


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Encoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer.num_layers):
            if i < 2:
                layer = Block(config, mm=True)
            else:
                layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states: torch.Tensor, text=None) -> torch.Tensor:
        for i, layer_block in enumerate(self.layer):
            if i == 2:
                hidden_states = torch.cat((hidden_states, text), 1)
                hidden_states = layer_block(hidden_states)
            elif i < 2:
                hidden_states, text = layer_block(hidden_states, text=text)
            else:
                hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class Block(nn.Module):
    def __init__(self, config: DictConfig, mm: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        if mm:
            self.att_norm_text = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm_text = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_text = Mlp(config)

        self.ffn = Mlp(config)
        self.attn = Attention(config, mm=mm)

    def forward(
        self, x: torch.Tensor, text: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if text is None:
            h = x
            x = self.attention_norm(x)
            x = self.attn(x)
            x = x + h

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x
        else:
            h = x
            h_text = text
            x = self.attention_norm(x)
            text = self.att_norm_text(text)
            x, text = self.attn(x, text)
            x = x + h
            text = text + h_text

            h = x
            h_text = text
            x = self.ffn_norm(x)
            text = self.ffn_norm_text(text)
            x = self.ffn(x)
            text = self.ffn_text(text)
            x = x + h
            text = text + h_text
            return x, text


class Attention(nn.Module):
    def __init__(self, config: DictConfig, mm: bool = True):
        super().__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        if mm:
            self.query_text = Linear(config.hidden_size, self.all_head_size)
            self.key_text = Linear(config.hidden_size, self.all_head_size)
            self.value_text = Linear(config.hidden_size, self.all_head_size)
            self.out_text = Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout_text = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_it = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_ti = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout_text = Dropout(config.transformer["attention_dropout_rate"])

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, text: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if text is not None:
            text_q = self.query_text(text)
            text_k = self.key_text(text)
            text_v = self.value_text(text)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if text is not None:
            query_layer_img = query_layer
            key_layer_img = key_layer
            value_layer_img = value_layer
            query_layer_text = self.transpose_for_scores(text_q)
            key_layer_text = self.transpose_for_scores(text_k)
            value_layer_text = self.transpose_for_scores(text_v)

        if text is None:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)
            attention_output = self.proj_dropout(attention_output)
            return attention_output
        else:
            attention_scores_img = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores_text = torch.matmul(query_layer_text, key_layer_text.transpose(-1, -2))
            attention_scores_it = torch.matmul(query_layer_img, key_layer_text.transpose(-1, -2))
            attention_scores_ti = torch.matmul(query_layer_text, key_layer_img.transpose(-1, -2))
            attention_scores_img = attention_scores_img / math.sqrt(self.attention_head_size)
            attention_probs_img = self.softmax(attention_scores_img)
            attention_probs_img = self.attn_dropout(attention_probs_img)

            attention_scores_text = attention_scores_text / math.sqrt(self.attention_head_size)
            attention_probs_text = self.softmax(attention_scores_text)
            attention_probs_text = self.attn_dropout_text(attention_probs_text)

            attention_scores_it = attention_scores_it / math.sqrt(self.attention_head_size)
            attention_probs_it = self.softmax(attention_scores_it)
            attention_probs_it = self.attn_dropout_it(attention_probs_it)

            attention_scores_ti = attention_scores_ti / math.sqrt(self.attention_head_size)
            attention_probs_ti = self.softmax(attention_scores_ti)
            attention_probs_ti = self.attn_dropout_ti(attention_probs_ti)

            context_layer_img = torch.matmul(attention_probs_img, value_layer_img)
            context_layer_img = context_layer_img.permute(0, 2, 1, 3).contiguous()
            context_layer_text = torch.matmul(attention_probs_text, value_layer_text)
            context_layer_text = context_layer_text.permute(0, 2, 1, 3).contiguous()
            context_layer_it = torch.matmul(attention_probs_it, value_layer_text)
            context_layer_it = context_layer_it.permute(0, 2, 1, 3).contiguous()
            context_layer_ti = torch.matmul(attention_probs_ti, value_layer_img)
            context_layer_ti = context_layer_ti.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer_img.size()[:-2] + (self.all_head_size,)
            context_layer_img = context_layer_img.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_text.size()[:-2] + (self.all_head_size,)
            context_layer_text = context_layer_text.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_it.size()[:-2] + (self.all_head_size,)
            context_layer_it = context_layer_it.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_ti.size()[:-2] + (self.all_head_size,)
            context_layer_ti = context_layer_ti.view(*new_context_layer_shape)
            attention_output_img = self.out((context_layer_img + context_layer_it) / 2)
            attention_output_text = self.out((context_layer_text + context_layer_ti) / 2)
            attention_output_img = self.proj_dropout(attention_output_img)
            attention_output_text = self.proj_dropout_text(attention_output_text)

            return attention_output_img, attention_output_text


class Mlp(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
