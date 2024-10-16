import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKV_TimeMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # Time weighting
        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        self.output = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        # Time weighting
        w = F.pad(self.time_w, (0, self.ctx_len - T))
        w = w.unsqueeze(0).unsqueeze(2)  # Add batch and head size dimensions
        w = torch.tile(w, (B, 1, T, T))  # Make sure w has the shape [B, n_head, T, T]
        w = w[:, :, :T, :T] * self.time_alpha * self.time_beta  # Now it has the right shape

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        sum_k = torch.cumsum(k, dim=1)
        kv = (k * v).view(B, T, self.n_head, self.head_size)

        # Use .reshape() instead of .view()
        wkv = torch.einsum('bhtu,buhc->bthc', w, kv).reshape(B, T, C)
        rwkv = torch.sigmoid(r) * wkv / sum_k

        return self.output(rwkv)



class RWKV_ChannelMix(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = 4 * config.n_embd  # Intermediate size
        self.key = nn.Linear(config.n_embd, hidden_size)  # Project to a higher dimension
        self.value = nn.Linear(config.n_embd, hidden_size)  # Same for value
        self.receptance = nn.Linear(config.n_embd, hidden_size)  # Adjust receptance to hidden_size
        self.output = nn.Linear(hidden_size, config.n_embd)  # Output back to original embedding size

    def forward(self, x):
        k = F.mish(self.key(x))  # Apply non-linearity to key projection
        v = self.value(x)  # Value projection
        r = torch.sigmoid(self.receptance(x))  # Receptance is now the same size as k and v

        # Element-wise multiplication of k, v, and r
        return self.output(k * v * r)


class RWKV_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.time_mix = RWKV_TimeMix(config)
        self.channel_mix = RWKV_ChannelMix(config)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x

class RWKV_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config in the model
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RWKV_Block(config) for _ in range(config.num_layers)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.head(x)






class RWKV_Config:
    def __init__(self, vocab_size, n_embd, ctx_len, num_layers, n_head):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.ctx_len = ctx_len
        self.num_layers = num_layers
        self.n_head = n_head




def train_rwkv(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            # Access vocab size from the model's output layer
            loss = criterion(outputs.view(-1, model.head.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")














from torch.utils.data import Dataset, DataLoader
import torch

class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens  # Tokens are passed directly, no need to call tokenizer again
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        input_seq = self.tokens[idx:idx + self.context_length]
        target_seq = self.tokens[idx + 1:idx + self.context_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)


# Simple tokenizer (can be replaced with sophisticated ones like Byte-Pair Encoding)
def simple_tokenizer(text):
    tokens = text.split()  # Split by whitespace
    vocab = {word: i for i, word in enumerate(set(tokens))}  # Create vocab
    tokenized_text = [vocab[word] for word in tokens]
    return tokenized_text, vocab


# Sample text data for training
text_data = "This is a simple language model example for testing the RWKV model training implementation."

# Tokenize the text and prepare dataset
tokens, vocab = simple_tokenizer(text_data)  # tokenizer now returns both tokenized text and vocab
context_length = 10  # How many tokens to consider as context for each training example

# Update TextDataset to receive only the tokenized text
dataset = TextDataset(tokens, context_length)

# Create the DataLoader
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)










# RWKV Model Training
config = RWKV_Config(vocab_size=len(vocab), n_embd=256, ctx_len=context_length, num_layers=6, n_head=8)
model = RWKV_Model(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


from torchvision.utils import save_image
print(model)

# Train the RWKV model
train_rwkv(model, train_dataloader, optimizer, criterion, epochs=10)



import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, vocab, prompt, max_new_tokens=50, context_length=10):
    model.eval()  # Set model to evaluation mode
    tokens = tokenizer(prompt)[0]  # Tokenize the input prompt
    generated = tokens[:]

    for _ in range(max_new_tokens):
        # Ensure the context length doesn't exceed the model's limit
        context_tokens = generated[-context_length:]

        # Convert tokens to tensor and pass through the model
        input_tensor = torch.tensor(context_tokens).unsqueeze(0)  # Add batch dimension
        logits = model(input_tensor)  # Get logits for next token

        # Take the logits for the last token in the sequence
        next_token_logits = logits[0, -1, :]

        # Option 1: Greedy decoding (choose token with highest probability)
        next_token = torch.argmax(next_token_logits).item()

        # Option 2: Sampling (if you want more diverse outputs, uncomment)
        # next_token_probs = F.softmax(next_token_logits, dim=-1)
        # next_token = torch.multinomial(next_token_probs, num_samples=1).item()

        # Append the predicted token to the generated sequence
        generated.append(next_token)

        # Stop generation if <EOS> token is generated (optional, if you have EOS)
        # if next_token == vocab['<EOS>']:
        #     break

    # Convert generated tokens back to text using the vocabulary
    inv_vocab = {v: k for k, v in vocab.items()}  # Inverse the vocab dictionary
    generated_text = ' '.join([inv_vocab[token] for token in generated])

    return generated_text

# Example usage:

prompt = "This is an example of"
generated_text = generate_text(model, simple_tokenizer, vocab, prompt, max_new_tokens=20)
print(generated_text)


