import cv2
import numpy as np
import os
from Image_Label import dataset_map


def create_multiclass_mask(data_entry):
    """
    Handles a single entry from dataset_map.
    Example input: dataset_map['0003b0bf24238450']
    """
    image_path = data_entry['image'] #Takes image path
    mask_info_list = data_entry['masks'] #Takes mask paths along with the labels

    #Load the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}") #Safety code
        return None, None

    h, w, _ = img.shape #Take the shape of the original image

    # Create blank canvas that we can build upon with heigh and width of the image and 3 RGB channels
    combined_mask = np.zeros((h, w,1), dtype=np.uint8)

    # Define color map (BGR)
    color_map = {
        'm03bk1': 1,  # Giraffe -> Class 1
        'm0bwd': 2,  # Elephant -> Class 2
        'm0c29q': 3,  # Leopard -> Class 3

    }

    #Iterate through the masks for a single image
    for mask_item in mask_info_list:
        mask_path = mask_item['path']
        label = mask_item['label']

        # Load binary mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            continue

        #Resize the mask to match the image
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Get color by label
        color = color_map.get(label.lower(), 0)

        # Apply color to the combined mask where the binary mask is active
        combined_mask[mask > 0] = color


    return img, combined_mask

#Define where I want to put these masks
base_output = os.getcwd()
images_dir = os.path.join(base_output, "images")
masks_dir = os.path.join(base_output, "masks")

processed_map = {}

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)


for image_id, entry in dataset_map.items():

    image, multi_mask = create_multiclass_mask(entry) #Image and combined mask

    if image is None:
        continue

    #Define where I want the files to be
    img_filename = f"{image_id}.jpg"
    mask_filename = f"{image_id}_mask.png"  #Keeping the mask in png format same as the original

    #Define full path name
    img_save_path = os.path.join(images_dir, img_filename)
    mask_save_path = os.path.join(masks_dir, mask_filename)

    #Writing the original image and the combined mask into their files
    original_img = cv2.imread(entry['image'])
    cv2.imwrite(img_save_path, original_img)
    cv2.imwrite(mask_save_path, multi_mask)

    #Storing the paths in the dictionary just to be safe
    processed_map[image_id] = {
        'image_path': img_save_path,
        'mask_path': mask_save_path,
        'labels': [m['label'] for m in entry['masks']]
    }
    #Print progress
    if len(processed_map) % 100 == 0:
        print(f"Saved {len(processed_map)}/6000...")

print(f"All files saved in: {os.path.abspath(base_output)}")