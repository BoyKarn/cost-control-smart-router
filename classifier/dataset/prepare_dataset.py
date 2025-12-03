from datasets import load_dataset

LABEL2ID = {
    "simple": 0,
    "medium": 1,
    "complex": 2
}

def load_and_prepare_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name)

    def encode_labels(example):
        label = example["label"]
        if label == "ultra_complex":
            label = "complex"
        example["label"] = LABEL2ID[label]
        return example
    dataset = dataset.map(encode_labels)
    return dataset


if __name__ == "__main__":
    dataset = load_and_prepare_dataset("akorn14/router-classifier-dataset")
    print(dataset)    