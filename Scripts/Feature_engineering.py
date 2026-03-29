import os
from pathlib import Path
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


image_dir = Path(os.getcwd()) / "images"
mask_dir = Path(os.getcwd()) / "masks"

image_mask_map = {}



#Iterate through the images and find matching masks
for img_path in image_dir.glob("*.jpg"):
    #Extract image name
    file_id = img_path.stem

    #Construct mask name I defined it as such
    mask_name = f"{file_id}_mask.png"
    full_mask_path = mask_dir / mask_name

    #Create image mask dictionary
    if full_mask_path.exists():
        image_mask_map[str(img_path)] = str(full_mask_path)



#Take all images
all_images = list(image_mask_map.keys())

#Split images into train/test
train_keys, test_keys = train_test_split(
    all_images,
    test_size=0.2,
    random_state=1
)

#Create different dictionaries for train and test
train_map = {k: image_mask_map[k] for k in train_keys}
test_map = {k: image_mask_map[k] for k in test_keys}



class MulticlassSegmentationDataset(Dataset):
    def __init__(self, image_mask_map, transform=None):
        self.image_paths = list(image_mask_map.keys()) #Image paths
        self.mask_paths = list(image_mask_map.values()) #Mask paths
        self.transform = transform #What transformations I want to do

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load RGB image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load RGB mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)


        # Apply Transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']



        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        #Convert mask to tensor and store them as integers (long)
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


train_transform = A.Compose([
    #Resize and rotate image and mask. Inter nearest doesn't blend the image for masks instead takes the nearest number
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR,mask_interpolation=cv2.INTER_NEAREST),
    A.Rotate(limit=35, p=0.2, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),

    #Transformation for better generalization
    A.GaussianBlur(blur_limit=(3, 7), p=0.05),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

], additional_targets={'mask': 'mask'})


test_transform = A.Compose([
    # Resize for testing so we can do tests
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
], additional_targets={'mask': 'mask'})

#Define train object
train_ds = MulticlassSegmentationDataset(
    image_mask_map=train_map,
    transform=train_transform
)

#Load in batches
train_loader = DataLoader(
    train_ds,
    batch_size=8,        # Number of images per step
    shuffle=True,        # Mixes the data every epoch
    num_workers=2,       # Uses multi-processing to load data faster

)

#Define test object
test_ds = MulticlassSegmentationDataset(
    image_mask_map=test_map,
    transform=test_transform

)

test_loader = DataLoader(
    test_ds,
    batch_size=8,        # Number of images per step
    shuffle=False,        # Mixes the data every epoch
    num_workers=4,       # Uses multi-processing to load data faster

)






