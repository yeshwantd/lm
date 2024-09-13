import torch
from torch.nn import Embedding, Module
import torch.nn.functional as F

class BigramModel(Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        logits = self.token_embedding_table(idx)
        return logits