import json
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_metric
from transformers import LayoutLMv2ForTokenClassification, Trainer, TrainingArguments
from transformers import LayoutLMv2Processor
import numpy as np
from PIL import Image
import evaluate  # Updated import for metrics

label2id = {
    "O": 0,  # Outside any named entity
    "Name": 1,
    "Father_s_Name": 2,
    "DOB": 3,
    "Pancard_number": 4,
}

id2label = {
    0: "O",  # Outside any named entity
    1: "Name",
    2: "Father_s_Name",
    3: "DOB",
    4: "Pancard_number",
}

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


json_file = r"K:\PROJECTS\COMPUTER-VISION-PROJECTS\CV01-PAN-CARD-TAMPERING-DETECTION\data\annotated_layoutLM\project-4-at-2024-08-30-16-01-8a139649.json"

with open(json_file, "r") as f:
    data = json.load(f)

all_words = []
all_boxes = []
all_word_labels = []
all_images = []
for item in data:
    tokens = item['transcription']  # Assuming this is a list of strings
    bboxes = [[int(b['x']), int(b['y']), int(b['x'] + b['width']), int(b['y'] + b['height'])] for b in item['bbox']]
    labels = [label['labels'][0] if isinstance(label['labels'], list) else label['labels'] for label in item['label']]
    label_ids = [label2id[label] for label in labels]
    path = "../../data/detected_pancards/" + item['ocr'].split("-")[-1]
    img = Image.open(path).convert("RGB")
    width, height = img.size
    bboxes = [normalize_bbox(i, width, height) for i in bboxes]
    all_words.append(tokens)
    all_boxes.append(bboxes)
    all_word_labels.append(label_ids)
    all_images.append(img)

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
encodings = []
for words, images, boxes, word_labels in zip(all_words, all_images, all_boxes, all_word_labels):
    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, padding="max_length", truncation=True)
    encoding['image'] = encoding['image'][0]
    encodings.append(encoding)

train_dataset = Dataset.from_list(encodings)
train_dataset.set_format(type="torch")
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                         num_labels=len(label2id)).to(device)

# Set id2label and label2id
model.config.id2label = id2label
model.config.label2id = label2id

# Metrics
metric = evaluate.load("seqeval", trust_remote_code=True)
return_entity_level_metrics = True


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


class ModelTrainer(Trainer):
    def get_train_dataloader(self):
        return train_dataloader


# Setting up the training arguments
args = TrainingArguments(
    output_dir="layoutlmv2-finetuned-v1",  # name of directory to store the checkpoints
    max_steps=1000,  # we train for a maximum of 1,000 batches
    warmup_ratio=0.1,  # we warmup a bit
    fp16=True  # we use mixed precision (less memory consumption)
)

# Initialize our Trainer
trainer = ModelTrainer(
    model=model,
    args=args,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Define the directory where you want to save the model
output_dir = "layoutlmv2-finetuned-v1"

# Save the trained model and tokenizer
trainer.save_model(output_dir)

print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    pass