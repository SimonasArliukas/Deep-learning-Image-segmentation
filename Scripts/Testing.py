from Feature_engineering import test_loader
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from Model import BetterUNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("mps")

#Load the architecture
model = BetterUNet(num_classes=4).to(device)

# Load the model
checkpoint = torch.load('attention_res_network_turbo.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


##Calculate Class-wise precision, recall and F1 (accuracy is not particularly useful)
def evaluate_segmentation(model, test_loader, device):
    model.eval()
    all_preds = []
    all_masks = []
    class_names = ['Background', 'Elephant', 'Leopard', 'Giraffe']

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    #Per class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_masks, all_preds, average=None, labels=[0, 1, 2, 3]
    )

    print(f"\n--- Per-Class Analysis ---")
    header = f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Pixels':<10}"
    print(header)
    print("-" * len(header))

    for i in range(len(class_names)):

        print(f"{class_names[i]:<15} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {f1[i]:<10.4f} | {support[i]:<10}")
    # Generate Confusion Matrix
    cm = confusion_matrix(all_masks, all_preds, labels=[0, 1, 2, 3])
    # Normalize by row (True Labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalised by True Class)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return f1  # Returning F1 array for further analysis

evaluate_segmentation(model, test_loader, device)

#Convert into colors for better visualization
def label_to_rgb(mask, palette):

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(palette):
        rgb[mask == class_id] = color

    return rgb

#Plot the batch
def save_batch_predictions(model, device, epoch, batch_index=0):
    palette = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]
    ], dtype=np.uint8)

    model.eval()
    loader_iter = iter(test_loader)

    for _ in range(batch_index + 1):
        images, masks = next(loader_iter)

    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        output = model(images)
        preds = torch.argmax(output, dim=1)

    batch_size = images.shape[0]

    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    for i in range(batch_size):

        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # Ensure values are 0-1 for plotting

        gt_color = palette[masks[i].cpu().numpy()]
        pred_color = palette[preds[i].cpu().numpy()]

        # Plot row by row
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Batch {batch_index} - Img {i}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title("U-Net Prediction")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"batch_{batch_index}_epoch_{epoch}.png")
    plt.close()


# Example: Plot the 3rd batch (index 2)
save_batch_predictions(model, device, 14, batch_index=4)
