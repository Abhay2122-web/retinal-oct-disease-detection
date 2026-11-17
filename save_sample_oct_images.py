"""
Save one real OCT image per class from the project's Hugging Face dataset.

Run from the project folder (the folder containing data_utils.py):
    python save_sample_oct_images.py

Images are saved in sample_oct_images/.
"""
from pathlib import Path
from data_utils import load_oct_dataset, get_class_names


def main():
    output_dir = Path("sample_oct_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OCT dataset...")
    dataset = load_oct_dataset()
    train_data = dataset["train"]
    class_names = get_class_names(train_data)

    saved_labels = set()
    print("Finding one image from each class...")

    for row in train_data:
        label = int(row["label"])
        if label in saved_labels:
            continue

        image = row["image"]
        if not hasattr(image, "save"):
            raise TypeError(
                "The dataset image is not a PIL image. "
                "Check the image conversion in data_utils.py."
            )

        class_name = str(class_names[label]).replace("/", "_")
        output_path = output_dir / f"{class_name}.png"
        image.convert("RGB").save(output_path)
        saved_labels.add(label)
        print(f"Saved: {output_path}")

        if len(saved_labels) == len(class_names):
            break

    if len(saved_labels) == len(class_names):
        print(f"Done. OCT samples saved in: {output_dir.resolve()}")
        print("Upload one of these PNG files in your Streamlit app.")
    else:
        print("Warning: not all classes were found in the training split.")


if __name__ == "__main__":
    main()
