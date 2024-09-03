from transformers import LayoutLMv2ForTokenClassification, set_seed, LayoutLMv2Processor
from PIL import Image
from datasets import load_dataset
import pytesseract
import torch
from transformers import BertTokenizerFast

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]



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

labels = label2id.keys()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model and processor
model = LayoutLMv2ForTokenClassification.from_pretrained("layoutlmv2-finetuned-v1",  num_labels=len(labels)).to(device)
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)

# Sample OCR data
path = r"K:\PROJECTS\COMPUTER-VISION-PROJECTS\CV01-PAN-CARD-TAMPERING-DETECTION\data\detected_pancards\2_skew_corrected_detected_pancard_0.jpg"
img = Image.open(path).convert('RGB') # Ensure image is a PIL Image
width, height = img.size


# Use pytesseract to get OCR data
data = pytesseract.image_to_data(img)
bounding_box = []
words = []


# Parse the OCR data
for i in data.split('\n')[1:]:
    res = i.split('\t')[-6:]
    if res[-1].strip() != '':
        words.append(res[-1])
        bb = [int(i) for i in res[:-2]]
        bb[2] += bb[0]
        bb[3] += bb[1]
        bounding_box.append(bb)
print(bounding_box)
bounding_box = [normalize_bbox(i, width, height) for i in bounding_box]
print(bounding_box)
encoding = processor(img, words, boxes=bounding_box, return_tensors="pt").to(device)


# input_ids = encoding["input_ids"].to(device)
# bbox = encoding["bbox"].to(device)
# image = encoding["image"].to(device)
# attention_mask = encoding["attention_mask"].to(device)


outputs = model(**encoding)
logits, loss = outputs.logits, outputs.loss
predicted_token_class_ids = logits.argmax(-1)
predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]
print(words)
print(predicted_tokens_classes)


# Initialize a compatible tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Get the offset mapping from the tokenizer
encoding_with_offsets = tokenizer(words, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
offset_mapping = encoding_with_offsets["offset_mapping"]


def align_labels_with_tokens(predictions, offset_mapping):
    aligned_predictions = []
    prev_word_start = None

    for idx, offset in enumerate(offset_mapping):
        if offset[0] == prev_word_start:
            # If it's the continuation of a word, skip it
            continue
        aligned_predictions.append(predictions[idx])
        prev_word_start = offset[0]

    return aligned_predictions


# Apply the function after making predictions
aligned_predictions = align_labels_with_tokens(predicted_tokens_classes, offset_mapping)


for i, j in zip(words, aligned_predictions):
    print(f"{i} - {j}")