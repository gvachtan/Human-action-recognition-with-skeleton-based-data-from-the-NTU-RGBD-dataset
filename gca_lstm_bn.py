import gdown
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the path to your numpy files
path = "the path to your data"
files = glob.glob(os.path.join(path, '*.npy'))

# Define a regular expression to extract numeric values
pattern = re.compile(r'\d+\.?\d*')

# The average number of frames is 80, the max number is 300
frame_count = 165
joint_count = 25
joint_dim = 3

# Process the skeleton data
data_processed = []
targets = []
subject_ids = []
camera_ids = []

for file_path in files:
    file_data = np.load(file_path, allow_pickle=True).item()

    # Extract the subject and camera IDs from the file name
    subject_id = int(re.search(r'P(\d+)', file_path).group(1))
    camera_id = int(re.search(r'C(\d+)', file_path).group(1))

    for actor_id in range(2):  # process data for up to two actors
        skeleton_key = 'skel_body' + str(actor_id)

        # Generate the target for the current file
        action_class = int(re.search(r'A(\d+)', file_path).group(1)) - 1

        # Initialize a variable to track if we've processed any data for this actor
        actor_data_processed = False

        if skeleton_key in file_data:
            skeleton_data = file_data[skeleton_key]
            skeleton_data_padded = skeleton_data[:frame_count, :] if skeleton_data.shape[0] >= frame_count else np.vstack([skeleton_data, np.zeros((frame_count - skeleton_data.shape[0], joint_count, joint_dim))])
            data_processed.append(skeleton_data_padded)
            actor_data_processed = True

        if actor_data_processed:
            targets.append(action_class)
            subject_ids.append(subject_id)
            camera_ids.append(camera_id)

data_processed = np.array(data_processed)
data_processed = data_processed.reshape(data_processed.shape[0], data_processed.shape[1], -1)

# Define the number of action classes
action_classes = len(set(targets))

# Create one-hot encoded targets
targets_one_hot = np.eye(action_classes)[targets]

# Convert sequences and targets to PyTorch tensors
sequences_tensor = torch.tensor(data_processed)
targets_tensor = torch.tensor(targets_one_hot)

# Define the datasets
dataset = data_utils.TensorDataset(sequences_tensor, targets_tensor)

# Define the data loaders
batch_size = 32

# Cross-View Evaluation
training_camera_ids = [2, 3]
train_indices = [i for i, camera_id in enumerate(camera_ids) if camera_id in training_camera_ids]
test_indices = [i for i, camera_id in enumerate(camera_ids) if camera_id not in training_camera_ids]

train_dataset = data_utils.Subset(dataset, train_indices)
test_dataset = data_utils.Subset(dataset, test_indices)

train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, value, key, query):
        N = query.shape[0]

        value_time, key_time, query_time = value.shape[1], key.shape[1], query.shape[1]

        # Transform inputs to Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split the hidden size into multiple heads
        Q = Q.view(N, query_time, self.num_heads, self.head_dim)
        K = K.view(N, key_time, self.num_heads, self.head_dim)
        V = V.view(N, value_time, self.num_heads, self.head_dim)

        # Calculate the attention scores
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Calculate the context vector
        context_vector = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, V])
        context_vector = context_vector.reshape(N, query_time, -1)

        # Final linear layer
        output = self.fc_out(context_vector)

        return output

num_heads = 20 # or any other number

class GCA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads, dropout_prob=0.5):
        super(GCA_LSTM, self).__init__()
        self.hidden_size = hidden_size
        dropout_prob = 0 if num_layers == 1 else dropout_prob
        self.bn = nn.BatchNorm1d(75)  # Initialize with the number of features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.lstm_bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.attention_bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (batch_size, num_features, sequence_length)
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # Change the order of dimensions back to (batch_size, sequence_length, num_features)
        output, _ = self.lstm(x)
        output = output.permute(0, 2, 1)  # Change the order of dimensions to (batch_size, hidden_size, sequence_length)
        output = self.lstm_bn(output)
        output = output.permute(0, 2, 1)  # Change the order of dimensions back to (batch_size, sequence_length, hidden_size)
        output = self.attention(output, output, output)
        output = output.permute(0, 2, 1)  # Change the order of dimensions to (batch_size, hidden_size, sequence_length)
        output = self.attention_bn(output)
        output = output.permute(0, 2, 1)  # Change the order of dimensions back to (batch_size, sequence_length, hidden_size)
        output = torch.mean(output, dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return output

# Set the hyperparameters for the model
input_size = joint_count * joint_dim
hidden_size = 320
num_layers = 3
num_classes = action_classes
learning_rate = 0.0005
num_epochs = 60

# Create an instance of the GCA_LSTM RNN model
model = GCA_LSTM(input_size, hidden_size, num_layers, num_classes, num_heads, dropout_prob=0.5)
model = model.to(device)  # Move model to GPU

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Train the model
for epoch in range(0, num_epochs):

    # Training loop
    train_loss = 0.0
    model.train()
    for inputs, labels in train_dataloader:
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs.float())
        loss = criterion(outputs, torch.max(labels, 1)[1])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataloader.dataset)

    # Validation loop
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs.float())
            loss = criterion(outputs, torch.max(labels, 1)[1])
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_dataloader.dataset)

    # Print the loss after each epoch
    print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss))

    # Initialize empty lists to store the true and predicted labels
    y_true = []
    y_pred = []

    # Loop over the validation set and make predictions
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)

            # Append the true and predicted labels to the lists
            y_true.extend(torch.max(labels, 1)[1].tolist())
            y_pred.extend(predicted.tolist())

    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    # Print the evaluation metrics
    print("Accuracy: {:.4f}, F1 Score: {:.4f}".format(accuracy, f1))

    # Step the scheduler
    scheduler.step(val_loss)
