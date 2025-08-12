import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# 1. Data Collection and Preprocessing
# A small, hypothetical dataset for demonstration
texts = [
    "This movie is fantastic!",
    "I really enjoyed the film.",
    "The acting was terrible.",
    "What a waste of time.",
    "Highly recommended.",
    "So boring and dull.",
    "Great performance by the lead.",
    "Couldn't stand the plot."
]
labels = [1, 1, 0, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

# Simple text cleaning and tokenization
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    return text.split()

tokenized_texts = [tokenize(text) for text in texts]

# Build vocabulary
word_counts = Counter()
for tokens in tokenized_texts:
    word_counts.update(tokens)

# Assign unique index to each word
vocab = {"<unk>": 0, "<pad>": 1} # Special tokens for unknown words and padding
for word, _ in word_counts.most_common():
    if word not in vocab:
        vocab[word] = len(vocab)

# Convert tokens to numerical IDs
def text_to_sequence(tokens, vocab, max_len):
    sequence = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    # Pad sequences to a fixed length
    if len(sequence) < max_len:
        sequence = sequence + [vocab["<pad>"]] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

max_sequence_length = max(len(t) for t in tokenized_texts) # Determine max length
sequences = [text_to_sequence(tokens, vocab, max_sequence_length) for tokens in tokenized_texts]

# Convert to PyTorch tensors
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Define the Model Architecture (LSTM)
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 for bidirectional LSTM
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text):
        # text = [batch size, sequence len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sequence len, embedding dim]

        output, (hidden, cell) = self.lstm(embedded)
        # output = [batch size, sequence len, hidden dim * num directions]
        # hidden = [num layers * num directions, batch size, hidden dim]

        # Use the concatenated hidden states from forward and backward LSTMs for the last layer
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = [batch size, hidden dim * num directions]

        return self.fc(hidden)

# Hyperparameters
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2 # Binary classification (positive/negative)
NUM_LAYERS = 2
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 10

model = TextClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_RATE)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 3. Choose Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to calculate accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.softmax(preds, dim=1)[:, 1])
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# 4. Training Loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for text, label in loader:
        text, label = text.to(device), label.to(device)

        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)

# Evaluation Loop
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for text, label in loader:
            text, label = text.to(device), label.to(device)
            predictions = model(text)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)

print("Starting training...")
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

print("Training complete.")

# Make predictions on a new sentence
def predict_sentiment(model, sentence, vocab, max_len, device):
    model.eval()
    tokenized = tokenize(sentence)
    indexed = text_to_sequence(tokenized, vocab, max_len)
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0) # Add batch dimension

    prediction = model(tensor)
    probability = torch.softmax(prediction, dim=1)
    predicted_class = torch.argmax(probability, dim=1).item()
    return "Positive" if predicted_class == 1 else "Negative", probability[0][predicted_class].item()

new_sentence = "This is an amazing product!"
sentiment, prob = predict_sentiment(model, new_sentence, vocab, max_sequence_length, device)
print(f"Sentence: '{new_sentence}' | Predicted Sentiment: {sentiment} (Confidence: {prob:.2f})")

new_sentence_2 = "I totally hate this."
sentiment, prob = predict_sentiment(model, new_sentence_2, vocab, max_sequence_length, device)
print(f"Sentence: '{new_sentence_2}' | Predicted Sentiment: {sentiment} (Confidence: {prob:.2f})")
