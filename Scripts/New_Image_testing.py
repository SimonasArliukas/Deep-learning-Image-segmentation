import torch
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path


def predict_new_images(model_path, image_folder, device, num_classes=4):

    # Resize and Normalize to keep data distribution identical to training
    transform = A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


    model = BetterUNet(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Extract state dict from your checkpoint dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    image_exts = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(list(Path(image_folder).glob(ext)))

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    for img_path in image_paths:

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]


        augmented = transform(image=image)
        input_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1).float()
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

        #  Inference
        with torch.no_grad():
            output = model(input_tensor)
            # Take the class with highest probability for each pixel
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()


        mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # Visualization
        show_prediction(image, mask_resized, img_path.name)


def show_prediction(img, mask, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original: {title}")
    plt.axis('off')

    plt.subplot(1, 2, 2)

    plt.imshow(mask, cmap='tab10', vmin=0, vmax=3)
    plt.title("Predicted Segmentation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Setup Device
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")


    MODEL_WEIGHTS = "best_attention_res_model.pth"
    NEW_IMAGES_DIR = "/Users/simonasarliukas/Desktop/Image Segmentation/Web_images"

    predict_new_images(MODEL_WEIGHTS, NEW_IMAGES_DIR, dev)