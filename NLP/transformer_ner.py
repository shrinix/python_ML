import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Data generation
FIELDS = {
    "id": lambda: ''.join(random.choices("0123456789", k=10)),
    "currency": lambda: random.choice(["USD", "EUR", "JPY", "GBP", "YEN"]),
    "amount": lambda: f"{random.randint(1,999999):010d}"
}
FIELD_TAGS = {k: k for k in FIELDS}

def generate_sample():
    order = random.sample(list(FIELDS), len(FIELDS))
    s, tags = "", []
    for field in order:
        val = FIELDS[field]()
        s += val
        tags += [f"B-{field}"] + [f"I-{field}"] * (len(val)-1)

    return s, tags

ALL_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
char2idx = {ch: i+1 for i, ch in enumerate(ALL_CHARS)}
idx2char = {i: ch for ch, i in char2idx.items()}
all_tags = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ["B", "I"]]
tag2idx = {tag: i for i, tag in enumerate(all_tags)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

def encode_sample(seq, tags, max_len=32):
    x = [char2idx.get(c, 0) for c in seq]
    y = [tag2idx[t] for t in tags]
    pad_len = max_len - len(x)
    return x + [0]*pad_len, y + [tag2idx["O"]]*pad_len

class SequenceDataset(Dataset):
    def __init__(self, n=1000, max_len=32):
        data = [generate_sample() for _ in range(n)]
        self.samples = [s for s, t in data]
        self.tags = [t for s, t in data]
        self.max_len = max_len
         #generate 10 random integers
        random_integers = [str(random.randint(0, 9)) for _ in range(10)]
        # Print 10 rndom samples and tags for inspection
        for i in random_integers:
            idx = int(i)
            sample, sample_tags = self.samples[idx], self.tags[idx]
            print("Sample:", sample)
            print("Tags:  ", sample_tags)
            print()
        input("Press Enter to continue...")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = encode_sample(self.samples[idx], self.tags[idx], self.max_len)
        return torch.tensor(x), torch.tensor(y)

class CharTransformerNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, d_model=128, nhead=4, nlayers=2, max_len=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.fc = nn.Linear(d_model, tagset_size)
    def forward(self, x):
        x = self.emb(x) + self.pos[:, :x.size(1)]
        x = self.encoder(x)
        return self.fc(x)

def train(model, loader, opt, loss_fn, epochs=5):
    for e in range(epochs):
        model.train()
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out.view(-1, out.size(-1)), yb.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {e+1}: Loss = {total / len(loader):.4f}")

def predict(model, seq, max_len=32):
    model.eval()
    x, _ = encode_sample(seq, ["O"]*len(seq), max_len=max_len)
    with torch.no_grad():
        logits = model(torch.tensor([x]).to(device))
        preds = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
    return list(zip(seq, [idx2tag[i] for i in preds[:len(seq)]]))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = SequenceDataset(2000)
    loader = DataLoader(data, batch_size=32, shuffle=True)
    model = CharTransformerNER(len(char2idx)+1, len(tag2idx)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train(model, loader, opt, loss_fn, epochs=10)

    test_seq, test_tags = generate_sample()
    print("Test Sequence:", test_seq)
    print("Expected Tags:", test_tags)
    print("Predicted Tags:")
    pred = predict(model, test_seq)
    pred_chars = [ch for ch, tag in pred]
    pred_tags = [tag for ch, tag in pred]
    print("Predicted Characters:", pred_chars)
    print("Predicted Tags:", pred_tags)
    # Compare expected and predicted tags
    for i, (exp, pred_tag) in enumerate(zip(test_tags, pred_tags)):
        print(f"Char: {test_seq[i]} | Expected: {exp} | Predicted: {pred_tag}")
