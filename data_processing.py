
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

"""
Includes: 
- build_pair_df
- class PairwiseImageDataset
- get_transform
- prepare_data
"""

def build_pair_df(
        base_path=".",
        sample_n=None,  # None = use full dataset, or specify number for subset
        image_dir="images",
        random_state=42
):
    """
    Load studies/votes TSVs, filter to a safer city study, map image IDs to filepaths,
    drop rows with missing images, and return a pair_df.
    """
    # load studies tsv
    studies = pd.read_csv(os.path.join(base_path, "studies.tsv"), sep='\t')
    # extract id for safer_multicity study
    safer_id = studies[studies["study_name"] == "safer_multicity"]["_id"]

    # load votes tsv
    votes = pd.read_csv(os.path.join(base_path, "votes.tsv"), sep='\t')

    # filter votes to only safer_multicity study
    pair_df = votes[votes['study_id'] == safer_id.values[0]]  

    # select only columns needed 
    pair_df = pair_df[['left', 'right', 'choice']] 
    pair_df = pair_df[pair_df['choice'].isin(['left', 'right'])] # filter out equals and some gibberish in 'choice' column 

    # FOR TESTING: cut down dataset (if sample_n is provided and not None)
    if sample_n is not None and sample_n > 0:
        pair_df = pair_df.sample(sample_n, random_state=random_state).reset_index(drop=True)
        # print(f"Sampled {sample_n} pairs from the dataset.")
    # else:
        # print(f"Using full dataset with {len(pair_df)} pairs.")

    # map image ids to image filepaths
    id_to_filepath = {}
    skipped_filenames = []

    for fname in os.listdir("images"):
        # only allow valid image extensions
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        parts = fname.split("_")

        # valid filename must contain the ID in index 2
        if len(parts) < 3:
            skipped_filenames.append(fname)
            continue

        # example: parts[2] = "513d5fd6fdc9f03587003c45.JPG"
        id_part = parts[2].split(".")[0]  # strip extension
        id_to_filepath[id_part] = os.path.join("images", fname)

    # print(f"Loaded {len(id_to_filepath)} image mappings.")
    # print(f"Skipped {len(skipped_filenames)} files due to format issues.")

    missing_ids = []   # <- track missing images

    def id_to_path(image_id):
        if image_id in id_to_filepath:
            return id_to_filepath[image_id]
        else:
            missing_ids.append(image_id)
            return None  # signal missing

    pair_df["filepathA"] = pair_df["left"].apply(id_to_path)
    pair_df["filepathB"] = pair_df["right"].apply(id_to_path)

    # remove rows where A or B is missing
    before = len(pair_df)
    pair_df = pair_df.dropna(subset=["filepathA", "filepathB"]).reset_index(drop=True)
    after = len(pair_df)

    pair_df["label"] = (pair_df["choice"] == "left").astype(int) # 1 if left is safer, else 0

    # print(f"Dropped {before - after} rows due to missing images.")
    # print(f"Missing image IDs count: {len(missing_ids)}")

    return pair_df


# Ceate pairwise dataset class
class PairwiseImageDataset(Dataset): 
    """
    Custom dataset for pairwise images containing image pairs and labels.

    Expects DataFrame columns: filepathA, filepathB, label (0 or 1)
    """
    def __init__(self, df, transform=None): # dataframe with columns: filepathA, filepathB, label (0 or 1)
        self.df = df.reset_index(drop=True) # reset index
        self.transform = transform # image transformations

        self.y_labels = (1.0 - 2.0 * self.df['label'].values)

    def __len__(self): # length of dataset
        return len(self.df) # number of rows in dataframe

    def __getitem__(self, idx): # get item at index idx
        row = self.df.iloc[idx] # get row at index idx

        imgA = Image.open(row["filepathA"]).convert("RGB") # open image A
        imgB = Image.open(row["filepathB"]).convert("RGB") # open image B

        if self.transform: # apply transformations if any
            imgA = self.transform(imgA) # transform image A
            imgB = self.transform(imgB) # transform image B

        label = torch.tensor(row["label"]).long()   # convert 0/1 to tensor
        
        y = self.y_labels[idx]
        y = torch.tensor(y, dtype=torch.float32)

        return imgA, imgB, label, y # return images and label


# TRANSFORMATION FUNCTION
def get_transform():
    """
    Standard image transformation pipeline for ResNet.
    """
    return transforms.Compose([ 
        transforms.Resize((256, 256)), # resize images
        transforms.CenterCrop(224), # center crop
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize( # normalize for Resnet 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# PREPARE DATA
def prepare_data(pair_df, seed=42, batch_size=64):
    """
    Splits data into train/val/test sets and returns DataLoaders
    """
    transform = get_transform()

    # 1. Extract unique images
    all_images = np.unique(pair_df[["filepathA", "filepathB"]].values.flatten())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_images)  # shuffle images

    # 2. Train/val/test split (80/10/10)
    n = len(all_images)
    train_end = int(0.60 * n)
    val_end   = int(0.80 * n)

    # set images in separate groups to prevent data leakage (i.e. no set sees the same image, but within sets, duplication is allowed)
    train_imgs = set(all_images[:train_end])
    val_imgs   = set(all_images[train_end:val_end])
    test_imgs  = set(all_images[val_end:])

    # 3. Split pairs
    train_pairs = pair_df[pair_df["filepathA"].isin(train_imgs) & pair_df["filepathB"].isin(train_imgs)]
    val_pairs = pair_df[pair_df["filepathA"].isin(val_imgs) & pair_df["filepathB"].isin(val_imgs)]
    test_pairs = pair_df[pair_df["filepathA"].isin(test_imgs) & pair_df["filepathB"].isin(test_imgs)]

    print(len(train_imgs), len(val_imgs), len(test_imgs))

    # convert to Pytorch 
    train_dataset = PairwiseImageDataset(train_pairs, transform)
    val_dataset   = PairwiseImageDataset(val_pairs, transform)
    test_dataset  = PairwiseImageDataset(test_pairs, transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    return train_loader, val_loader, test_loader