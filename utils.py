import torch

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

def get_data(tokenizer):
    data = torch.tensor(tokenizer.encode(tokenizer.get_text()))
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]
    return train_data, test_data