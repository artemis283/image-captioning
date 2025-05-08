import math
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
import PIL
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, max_seq_len, num_heads=1, bias=False, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_size = head_size
        self.bias = bias
        self.dropout = dropout

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        """arguments: 
        embedding_dim = size of embedding dimension
        num_heads = number of attention heads
        max_seq_len = maximum sequence length
        bias = whether to use bias in the linear layer
        dropout = probability of dropout
        """

        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        self.attention_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().unsqueeze(0).unsqueeze(0))

    def forward(self, x): 
        batch_size, max_seq_len, _ = x.size() 

        # compute query, key and value vectors for all heads in a batch
        # split the embedding dimension into query, key and value
        Q, K, V = self.c_attn(x).split(self.embedding_dim, dim=2) # [batch_size, max_seq_len, embedding_dim]
        
        # reshape the query, key and value vectors to have a separate head for each token
        Q = Q.view(batch_size, max_seq_len, self.num_heads, self.head_size).transpose(1, 2) # [batch_size, max_seq_len, num_heads, head_size]
        K = K.view(batch_size, max_seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, max_seq_len, self.num_heads, self.head_size).transpose(1, 2)

        attention = (Q @ K.transpose(-2, -1)) * (1.0/math.sqrt(K.size(-1))) # transpose swaps the last two dimensions of K = (1,5,24) @ (1,24,5) = (1,5,5)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().unsqueeze(0).unsqueeze(0).to(x.device)
        attention = attention.masked_fill(~mask[:, :, :max_seq_len, :max_seq_len], float("-inf"))  
        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        hidden_state = attention @ V # [batch_size, num_heads, max_seq_len, head_size]

        hidden_state = hidden_state.transpose(1, 2).contiguous().view(batch_size, max_seq_len, self.embedding_dim)
        hidden_state = self.resid_dropout(hidden_state)

        return hidden_state
    

class FNN(nn.Module):

    def __init__(self, embedding_dim, bias=False, dropout=0.2):
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * embedding_dim, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x
    

# putting it all together
class DecoderBlock(nn.Module):

    def __init__(self, embedding_dim, head_size, max_seq_len, num_heads=1, bias=False, dropout=0.2):
        super().__init__()

        self.masked_attention = MaskedAttention(embedding_dim, head_size, max_seq_len, num_heads, bias, dropout)
        self.fnn = FNN(embedding_dim, bias, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.masked_attention(self.norm1(x))
        x = x + self.fnn(self.norm2(x))

        return x
    

class TransformerDecoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", embedding_dim=512, num_heads=8, max_seq_len=50, size_of_vocab=49408, num_layers=6, bias=False, dropout=0.2, head_size=64):
        super().__init__()

        self.clip_model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.embedding_dim = self.clip_model.config.hidden_size

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.size_of_vocab = len(self.tokenizer)
        self.bias = bias
        self.dropout = dropout
        self.head_size = head_size



        self.transformer = nn.ModuleDict(dict(
            dropout = nn.Dropout(dropout),
            blocks = nn.ModuleList([DecoderBlock(embedding_dim, head_size, max_seq_len, num_heads, bias, dropout) for _ in range(num_layers)]),
            layer_norm = nn.LayerNorm(embedding_dim),
            head = nn.Linear(embedding_dim, size_of_vocab, bias=bias)
        ))

    def forward(self, captions, targets=None):
        x = self.transformer['dropout'](captions)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.layer_norm(x)

        if targets is not None:
            # compute the loss if we are given targets
            logits = self.transformer['head'](x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        else:
            # only look at last token if performing inference
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss