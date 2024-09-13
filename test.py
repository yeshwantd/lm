from tokenizer import Tokenizer
from bigram import BigramModel
import torch
from utils import get_batch, get_data
import torch.nn.functional as F
import numpy as np

def test_tokenizer():
    tokenizer = Tokenizer("sherlock.txt")
    encoded_text = tokenizer.encode("sherlock holmes was a genius")
    decoded_text = tokenizer.decode(encoded_text)
    assert isinstance(encoded_text, list)
    assert decoded_text == "sherlock holmes was a genius"
    return tokenizer

def test_get_data(tokenizer):
    train_data, test_data = get_data(tokenizer)
    assert isinstance(train_data, torch.Tensor)
    assert isinstance(test_data, torch.Tensor)
    return train_data, test_data


def test_bigram_model(tokenizer, train_data):
    vocab_size = tokenizer.get_vocab_size()
    model = BigramModel(vocab_size)
    token_ids = tokenizer.encode("sherlock holmes was a genius")
    logits = model(torch.tensor(token_ids))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (len(token_ids), vocab_size)

    x, y = get_batch(train_data, 8, 4)
    logits = model(x).view(-1, vocab_size)
    y = y.view(-1)
    loss = F.cross_entropy(logits, y)
    print(f"Losses from model = {loss.item()} and from -np.log(1/vocab_size) = {-np.log(1/vocab_size)}")

if __name__ == "__main__":
    tokenizer = test_tokenizer()
    train_data, test_data = test_get_data(tokenizer)
    test_bigram_model(tokenizer, train_data)