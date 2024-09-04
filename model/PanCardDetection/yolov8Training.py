from ultralytics import YOLO
# import torch

# Load the YOLO model
model = YOLO("./yolov8n.pt")

# # Check if CUDA is available and set the device accordingly
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# Print the device being used for training
# print(f"Using device: {device}")

try:
    # Train the model
    result = model.train(data='../../data/annotated/ForPanCardDetection/data.yaml',
                         epochs=20, workers=0, amp=False)
    print("Training completed successfully!")

    # Export the model to ONNX format
    success = model.export(format='onnx')

    if success:
        print("Model exported to ONNX format successfully!")
    else:
        print("Model export to ONNX format failed!")

except Exception as e:
    print(f"An error occurred during training or exporting: {e}")
