import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os

# Check if a GPU is available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

WORD_SIZE = 13

# Load and process the text file
def load_and_process_words(file_path):
    vocab = []
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
        for word in words:
            word = word.lower()
            if 3 < len(word) <= WORD_SIZE:
                word = word.ljust(WORD_SIZE, '_')
                vocab.append(word)
    return vocab

# Character to number conversion
def char_to_num(char):
    return 0 if char == '_' else ord(char) - ord('a') + 1

# Number to character conversion
def num_to_char(num):
    return '_' if num == 0 else chr(num + ord('a') - 1)

# Word to tensor conversion
def word_to_numlist(word):
    numlist = [char_to_num(char) for char in word]
    return torch.tensor(numlist, dtype=torch.long).to(device)

# Tensor to word conversion
def numlist_to_word(numlist):
    return ''.join([num_to_char(num.item()) for num in numlist])

# Autocomplete model definition
class autocompleteModel(nn.Module):
    def __init__(self, alphabet_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding, LSTM, Fully connected layers
        self.embedding = nn.Embedding(alphabet_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, alphabet_size)
        
    def forward(self, character, hidden_state, cell_state):
        embedded = self.embedding(character)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded.view(1, 1, -1), (hidden_state, cell_state))
        output = self.fc(lstm_out.view(1, -1))
        return output, hidden_state, cell_state
    
    def initial_state(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        return (h0, c0)
    
    def trainModel(self, vocab, epochs=5, batch_size=100, learning_rate=0.005):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        loss_log = []
        
        for e in range(epochs):
            print(f"Epoch {e+1}")
            random.shuffle(vocab)
            
            for i in range(0, len(vocab), batch_size):
                vocab_batch = vocab[i:i + batch_size]
                optimizer.zero_grad()
                batch_loss = 0
                
                for word in vocab_batch:
                    hidden_state, cell_state = self.initial_state()
                    word_tensor = word_to_numlist(word)
                    input_chars = word_tensor[:-1]
                    target_chars = word_tensor[1:]
                    
                    for c in range(WORD_SIZE - 1):
                        input_char = input_chars[c].unsqueeze(0)
                        target_char = target_chars[c].unsqueeze(0)
                        output, hidden_state, cell_state = self.forward(input_char, hidden_state, cell_state)
                        loss = criterion(output, target_char)
                        batch_loss += loss
                
                batch_loss /= len(vocab_batch)
                batch_loss.backward()
                optimizer.step()
                loss_log.append(batch_loss.item())
            
            print(f"Epoch {e+1}, Loss: {batch_loss.item()}")

        return loss_log

    def autocomplete(self, sample):
        self.eval()
        completed_list = []
        for literal in sample:
            padded_literal = literal.ljust(WORD_SIZE, '_')
            input_tensor = word_to_numlist(padded_literal)
            hidden_state, cell_state = self.initial_state()
            predicted_word = literal

            for i in range(len(literal)):
                input_char = input_tensor[i].unsqueeze(0)
                output, hidden_state, cell_state = self.forward(input_char, hidden_state, cell_state)

            for i in range(len(literal), WORD_SIZE):
                probabilities = torch.softmax(output, dim=1)
                predicted_char = torch.multinomial(probabilities, 1).item()
                predicted_word += num_to_char(predicted_char)
                input_char = torch.tensor([predicted_char]).unsqueeze(0).to(device)
                output, hidden_state, cell_state = self.forward(input_char, hidden_state, cell_state)
            
            completed_list.append(predicted_word)
        
        return completed_list


# Training loop with different hyperparameter combinations
embedding_dims = [32, 64]
hidden_sizes = [64, 128]
learning_rates = [0.001, 0.005]
batch_sizes = [50, 100]
epoch_options = [10, 15, 20]

best_loss = float('inf')
best_model = None
best_hyperparams = {}
loss_curves = []


file_path = 'wordlist.txt'
vocab = load_and_process_words(file_path)

for embed_dim in embedding_dims:
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epoch_options:
                    model = autocompleteModel(alphabet_size=27, embed_dim=embed_dim, hidden_size=hidden_size, num_layers=1).to(device)
                    print(f"\nTraining with embed_dim={embed_dim}, hidden_size={hidden_size}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
                    loss_log = model.trainModel(vocab, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                    loss_curves.append((embed_dim, hidden_size, learning_rate, batch_size, epochs, loss_log))
                    final_loss = loss_log[-1]
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_model = model
                        best_hyperparams = {
                            'embed_dim': embed_dim,
                            'hidden_size': hidden_size,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }

# Plot and save loss curves
if not os.path.exists('plots'):
    os.makedirs('plots')

for embed_dim, hidden_size, lr, batch_size, epochs, loss_log in loss_curves:
    plt.plot(loss_log, label=f"Embed: {embed_dim}, Hidden: {hidden_size}, LR: {lr}, Batch: {batch_size}, Epochs: {epochs}")

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Hyperparameter Combinations')
plt.legend()
plt.savefig('plots/loss_curves.png')
plt.show()

# Display the best model hyperparameters
print(f"Best model with loss {best_loss:.4f}: {best_hyperparams}")
