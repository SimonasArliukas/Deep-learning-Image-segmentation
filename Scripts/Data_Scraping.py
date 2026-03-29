import fiftyone.zoo as foz

#Dowloads the data from openImages to the device in a specialized folder for faster processing
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["segmentations"],
    classes=["Elephant", "Leopard", "Giraffe"],
    max_samples=6000,
    cleanup=True,

)



