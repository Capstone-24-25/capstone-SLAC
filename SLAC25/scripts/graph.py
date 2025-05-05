import re
import matplotlib.pyplot as plt

# File path
log_file = "test_2025-05-01_16-40-08.out"

# Initialize lists to store losses
train_loss = []
test_loss = []

# Read file and extract loss values
with open(log_file, 'r') as f:
    for line in f:
        train_match = re.search(r"\[AutoEncoder Epoch \d+\] Train Loss: ([\d.]+)", line)
        test_match = re.search(r"\[AutoEncoder Epoch \d+\] Test Loss: ([\d.]+)", line)
        if train_match:
            train_loss.append(float(train_match.group(1)))
        if test_match:
            test_loss.append(float(test_match.group(1)))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AutoEncoder Training vs Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to file
plt.savefig("autoencoder_loss_plot.png", dpi=300)

# Show plot
plt.show()

