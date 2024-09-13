import tokenizer
import torch
from torch.optim import Adam
import torch.nn.functional as F
from bigram import BigramModel
import numpy as np
from utils import get_batch, get_data

def train_bigram_model(train_data, test_data, vocab_size, block_size, batch_size):
    model = BigramModel(vocab_size)
    optimizer = Adam(model.parameters(), lr=1e-1)
    epochs = 10
    batches = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        batch_losses = []
        for _ in range(batches):
            x, y = get_batch(train_data, block_size, batch_size)            
            logits = model(x).view(-1, vocab_size)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(sum(batch_losses) / len(batch_losses))
        print(f"Epoch: {epoch+1}, Loss: {train_losses[-1]}")

        if epoch % 10 == 0 and epoch != 0:
            with torch.no_grad():
                batch_test_losses = []
                for _ in range(100):
                    x, y = get_batch(test_data, block_size, batch_size)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    batch_test_losses.append(loss.item())
                test_losses.append(sum(batch_test_losses) / len(batch_test_losses))
                print(f"Test Loss: {batch_test_losses[-1]}")

    return train_losses, test_losses

if __name__ == "__main__":
    tokenizer = tokenizer.Tokenizer("sherlock.txt")
    train_data, test_data = get_data(tokenizer)
    
    vocab_size = tokenizer.get_vocab_size()
    batch_size = 4
    block_size = 8

    train_losses, test_losses = train_bigram_model(train_data, test_data, vocab_size, block_size, batch_size)
    

