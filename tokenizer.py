import re

class Tokenizer:
    def __init__(self, file):
        with open(file, mode='r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = re.sub(r'[^a-zA-Z\s]', '', self.text.lower())
            self.vocab = sorted(list(set(self.text.split())))
            self.token_to_id = dict()
            self.id_to_token = dict()
            for i, token in enumerate(self.vocab):
                self.token_to_id[token] = i
                self.id_to_token[i] = token
    
    def encode(self, text):
        return [self.token_to_id[token] for token in text.split()]
    
    def decode(self, tokens):
        return ' '.join([self.id_to_token[token] for token in tokens])

    def get_vocab(self):
        return self.vocab
    
    def get_text(self):
        return self.text
    
    def get_vocab_size(self):
        return len(self.vocab)