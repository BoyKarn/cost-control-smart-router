from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import evaluate

DATASET_NAME = "akorn14/router-classifier-dataset"
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

LABEL2ID = { "simple": 0, "medium": 1, "complex": 2}
ID2LABEL = { 0: "simple", 1: "medium", 2: "complex"}

def tokenize_fn(batch):
    return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=256)

if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

     tokenized = dataset.map(tokenize_fn, batched=True)
    
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
       logits , labels = eval_pred
       preds = logits.argmax(axis=-1)
       return accuracy.compute(predictions=preds, references=labels)

    args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       save_strategy="epoch",
       learning_rate=2e-5,
       per_device_train_batch_size=4,
       per_device_eval_batch_size=4,
       num_train_epochs=2,
       weight_decay=0.01,
       push_to_hub=False,
       logging_steps=20
    )

    trainer=Trainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=tokenized["train"],
       eval_dataset=tokenized["validation"],
       compute_metrics=compute_metrics,
       args=args
    )

    trainer.train()
    model.save_pretrained("./classifier_model")
    tokenizer.save_pretrained("./classifier_model")