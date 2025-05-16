import re
import matplotlib.pyplot as plt

# File path
log_file = "test_2025-05-01_16-40-08.out"

# Initialize lists to store values
train_loss = []
test_loss = []
classifier_loss = []
classifier_accuracy = []

# Read file and extract values
with open(log_file, 'r') as f:
    for line in f:
        train_match = re.search(r"\[AutoEncoder Epoch \d+\] Train Loss: ([\d.]+)", line)
        test_match = re.search(r"\[AutoEncoder Epoch \d+\] Test Loss: ([\d.]+)", line)
        clf_match = re.search(r"Epoch \[\d+/\d+\] Loss: ([\d.]+) Accuracy: ([\d.]+)", line)

        if train_match:
            train_loss.append(float(train_match.group(1)))
        if test_match:
            test_loss.append(float(test_match.group(1)))
        if clf_match:
            classifier_loss.append(float(clf_match.group(1)))
            classifier_accuracy.append(float(clf_match.group(2)))

# Plot autoencoder losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AutoEncoder Loss")
plt.legend()
plt.grid(True)

# Plot classifier accuracy
plt.subplot(1, 2, 2)
plt.plot(classifier_accuracy, label="Classifier Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy")
plt.ylim(0, 1)  # Accuracy from 0 to 1
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_summary_plot.png", dpi=300)
plt.show()
