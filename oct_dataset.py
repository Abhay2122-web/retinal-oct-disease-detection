"""Dataset inspection script, adapted from the supplied notebook."""
import matplotlib.pyplot as plt
from data_utils import load_oct_dataset, get_class_names

if __name__ == "__main__":
    ds = load_oct_dataset()
    train_data, test_data = ds["train"], ds["test"]
    names = get_class_names(train_data)
    print(ds)
    print("Training samples:", len(train_data))
    print("Testing samples:", len(test_data))
    print("Labels:", names)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, ax in enumerate(axes.flat):
        sample = train_data[i]
        ax.imshow(sample["image"], cmap="gray")
        ax.set_title(names[int(sample["label"])])
        ax.axis("off")
    plt.tight_layout()
    plt.show()
