import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from SLAC25.dataloader import DataLoaderFactory
from SLAC25.dataset import ImageDataset

# Create a temporary directory for testing
TEST_DIR = "/tmp/test_dataloader"
os.makedirs(TEST_DIR, exist_ok=True)

# Generate 10 fake images and labels
def create_fake_data():
    image_dir = os.path.join(TEST_DIR, "images")
    os.makedirs(image_dir, exist_ok=True)
    labels = []
    for i in range(10):
        # Create a fake image
        img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
        img_path = os.path.join(image_dir, f"image_{i}.png")
        img.save(img_path)
        # Create a fake label
        labels.append({"image_path": img_path, "label_id": i % 4})  

    # Save labels to a CSV file
    labels_df = pd.DataFrame(labels)
    labels_csv_path = os.path.join(TEST_DIR, "labels.csv")
    labels_df.to_csv(labels_csv_path, index=False)
    return labels_csv_path

# Mock training loop
def mock_training_loop(dataloader):
    for batch in dataloader:  
        images, labels = batch
        assert images.shape[0] > 0  # Ensure batch has data
        assert labels.shape[0] > 0
    print("Training loop executed successfully.")

# Test for random sampler
def test_random_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    factory.setRandomSampler()
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Random sampler test passed successfully.")

# Test for sequential sampler
def test_sequential_sampler():
    labels_csv_path = create_fake_data()
    dataset = ImageDataset(labels_csv_path)
    factory = DataLoaderFactory(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    factory.setSequentialSampler()
    dataloader = factory.outputDataLoader()
    mock_training_loop(dataloader)
    print("Sequential sampler test passed successfully.")

if __name__ == "__main__":
    test_random_sampler()
    test_sequential_sampler()