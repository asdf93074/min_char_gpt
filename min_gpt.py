import torch
import torch.nn as nn
import sys
from torch.nn import functional as F

device = torch.device('cuda')

data = open('tinyshakespeare.txt', 'r').read()
vocab = sorted(list(set(data)))
stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(data), dtype=torch.long)

#-----hyperparams---
block_size = 128 
batch_size = 64 
vocab_size = len(vocab) 
embd_size = 100 
learning_rate = 3e-4
num_heads = 4 
num_layers = 4
steps = 100000
sample_every = 1000
dropout = 0.2
#-------------------

data_train = data[:int(0.9*len(data))]
data_val = data[int(0.9*len(data)):]

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_size)
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out)) 

class FeedForward(nn.Module):
    def __init__(self, embd_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embd_size, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, embd_size // num_heads)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embd_size)
        self.pos_embedding = nn.Embedding(block_size, embd_size)
        self.blocks = nn.Sequential(*[Block(embd_size, num_heads) for _ in range(num_layers)])
        self.lin = nn.Linear(embd_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_embedding(idx) # B, T, C
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lin(x) # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
        
def generate(model, n):
    start = torch.zeros((1,1), dtype=torch.long, device=device)
    for _ in range(n):
        pred, loss = model(start[:, -block_size:])
        pred = pred[:,-1,:]
        probs = F.softmax(pred, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        start = torch.cat((start, idx_next), dim=1) 
    return start

def get_batch(split):
    data = data_train if split == 'train' else data_val
    offset = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in offset])
    y = torch.stack([data[i+1:i+block_size+1] for i in offset])

    return x.to(device), y.to(device)

def estimate_losses(m):
    losses = {}
    m.eval()
    for split in ['train', 'val']:
        loss_track = torch.zeros(200) 
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            loss_track[k] = loss.item()
        losses[split] = loss_track
    m.train()
    return losses

model = NN()
m = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
param_size = 0.0
buff_size = 0.0
for p in model.parameters():
    param_size += p.nelement() * p.element_size()
for b in model.buffers():
    buff_size += b.nelement() * b.element_size()
print((param_size + buff_size) / 1024, 'MB')

optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for s in range(steps): 
    X, Y = get_batch('train')
    
    logits, loss = model(X, Y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    if s % sample_every == 0:
        print('---------------------')
        losses = estimate_losses(model)
        print(f'iter: {s} train loss: {losses['train'].mean()} val loss: {losses['val'].mean()}')
        print('\n')
        print(decode(generate(model, 500)[0].tolist()))
        print('---------------------')

