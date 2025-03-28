import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')  # ran into error when using plt.show()
import matplotlib.pyplot as plt

# Configurable Parameters
hidden_size = 256  # Hidden layer size
batch_size = 32  # Batch size for DataLoader
num_epochs = 12  # Number of training epochs
learning_rate = 5e-4  # Learning rate for optimizer
dropout_rate = 0.5  # Dropout rate

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Load and Preprocess Data
def load_data(filepath):
    """Loads dataset from CSV, normalizes pixels, and splits into train/val/test sets."""
    data = pd.read_csv(filepath)

    # Extract features and labels
    X = data.drop(columns=["label"]).values.astype('float32') / 255.0  # Normalize pixels
    y = data["label"].values

    # Split into train (70%), validation (10%), and test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2 / 3, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Load dataset
X_train, X_val, X_test, y_train, y_val, y_test = load_data("Assignment5.csv")


# print(f"Training data: {X_train.shape}, {y_train.shape}")
# print(f"Validation data: {X_val.shape}, {y_val.shape}")
# print(f"Testing data: {X_test.shape}, {y_test.shape}")

# 3. Convert Data to PyTorch Tensors and DataLoaders
def create_dataloader(X, y, batch_size=batch_size, shuffle=True):
    """Creates a PyTorch DataLoader from NumPy arrays."""
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = create_dataloader(X_train, y_train, batch_size)
val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)
test_loader = create_dataloader(X_test, y_test, batch_size, shuffle=False)


# 4. Define the ANN Model
class ANN(nn.Module):
    def __init__(self, hidden_size=hidden_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)  # Input (28x28 images) -> Hidden Layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden Layer 1 -> Hidden Layer 2
        self.fc3 = nn.Linear(hidden_size, 10)  # Output layer (0-9 digits)
        self.dropout = nn.Dropout(dropout_rate)  # Use configurable dropout rate

    def forward(self, x):
        # Relu activation for hidden layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # CrossEntropyLoss applies softmax automatically
        return x


# Initialize model
model = ANN(hidden_size).to(device)


def train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate):
    """Trains a model using the given data loaders and prints validation accuracy each epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # adjust learning rate dynamically

    val_accuracy_list = []  # For validation accuracy tracking

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Train the model: forward pass
            loss = criterion(outputs, labels)  # Calculate error (CrossEntropyLoss): based on predicted vs actual
            optimizer.zero_grad()  # Zero out gradients from previous iteration
            loss.backward()  # Backpropagation: compute gradients
            optimizer.step()  # Update weights

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Check
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracy_list.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

    # Save your trained model weights
    torch.save(model.state_dict(), "trained_ann.pth")
    return val_accuracy_list


def eval_model(model, test_loader):
    """Evaluates the model on the test set."""
    # Load trained weights
    model.load_state_dict(torch.load("trained_ann.pth"))
    model.eval()

    correct, total = 0, 0
    num_classes = 10
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                total_per_class[label] += 1
                if label == prediction:
                    correct_per_class[label] += 1

    total_accuracy = 100 * correct / total
    print(f"\nOverall Test Accuracy: {total_accuracy:.2f}%")

    for i in range(num_classes):
        acc = 100 * correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0.0
        print(f"Accuracy for class {i}: {acc:.2f}%")


val_acc_history = train_model(model, train_loader, val_loader,
                              num_epochs=num_epochs,
                              learning_rate=learning_rate)

eval_model(model, test_loader)

epochs = range(1, len(val_acc_history) + 1)
plt.plot(epochs, val_acc_history, marker='o')
plt.title(f'Validation Accuracy Over Epochs (Final: {val_acc_history[-1]:.2f}%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig("val_accuracy_plot.png", dpi=300)
