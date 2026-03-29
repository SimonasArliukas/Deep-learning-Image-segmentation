import os
from collections import defaultdict

images_path = "/Users/simonasarliukas/fiftyone/open-images-v7/train/data"
labels_base_path = "/Users/simonasarliukas/fiftyone/open-images-v7/train/labels/masks"

mask_index = defaultdict(list)

print("Indexing masks... this happens once.")

for shard in os.listdir(labels_base_path):
    shard_path = os.path.join(labels_base_path, shard)

    if os.path.isdir(shard_path):
        for mask_file in os.listdir(shard_path):
            if mask_file.endswith('.png'):
                # Extract image_id: 'd4a123_m02pv19_1.png' -> 'd4a123'
                image_id = mask_file.split('_')[0]
                mask_index[image_id].append(os.path.join(shard_path, mask_file))


dataset_map = {}
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg'))]

print(f"Mapping {len(image_files)} images...")
for img_file in image_files:
    image_id = os.path.splitext(img_file)[0]


    matching_masks = mask_index.get(image_id)

    if matching_masks:
        mask_details = []
        for mask_path in matching_masks:
            mask_filename = os.path.basename(mask_path)
            parts = mask_filename.split('_')
            label = parts[1] if len(parts) > 1 else "unknown"

            mask_details.append({
                'path': mask_path,
                'label': label
            })

        dataset_map[image_id] = {
            'image': os.path.join(images_path, img_file),
            'masks': mask_details
        }

print(f"Done! Successfully mapped {len(dataset_map)} images.")

