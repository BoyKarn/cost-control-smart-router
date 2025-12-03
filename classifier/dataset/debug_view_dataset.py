from datasets import load_dataset

dataset = load_dataset("akorn14/router-classifier-dataset")

train_labels = set(example["label"] for example in dataset["train"])
valid_labels = set(example["label"] for example in dataset["validation"])
test_labels = set(example["label"] for example in dataset["test"])

print("Train labels: ", train_labels)
print("Validation labels: ", valid_labels)
print("Test labels: ", test_labels)