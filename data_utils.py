"""Load and prepare the OCT2017 dataset hosted on Hugging Face."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from config import DATASET_ID, IMAGE_SIZE

def load_oct_dataset():
    """Downloads/caches the public Hugging Face OCT dataset."""
    dataset = load_dataset(DATASET_ID)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Expected Hugging Face dataset splits named 'train' and 'test'.")
    return dataset

def make_transforms(image_size=IMAGE_SIZE, training=False):
    ops = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]
    if training:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)

class HuggingFaceOCTDataset(Dataset):
    def __init__(self, hf_split, indices=None, image_size=IMAGE_SIZE, training=False):
        self.split = hf_split
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        self.transform = make_transforms(image_size, training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.split[self.indices[index]]
        image = row["image"].convert("RGB")
        label = int(row["label"])
        return self.transform(image), label

def get_class_names(hf_split):
    names = hf_split.features["label"].names
    # Keep the dataset's own class ordering, rather than assuming alphabetical order.
    return [str(name).upper() for name in names]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
