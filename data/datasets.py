"""
This file contains a class specifically for importing the Caltech 200 (2011) bird
species dataset. The class contains methods for downloading, verifying, loading,
and preprocessing the dataset.

Author: Jordan Miller

Sources:
    [1] https://www.vision.caltech.edu/datasets/cub_200_2011/
    [2] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
import tarfile
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

# This class inherits from torch.utils.data.Dataset, making it compatible with
# PyTorch's DataLoader for batch processing as per [2]
class Cub2011(Dataset):
    
    # As per [1] we select the CUB 200 (2011) dataset.
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
 
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        """Constructor for the Cub2011 Dataset class."""
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

    def _load_metadata(self):
        """Loads filenames, labels, and splits."""
        images_path = os.path.join(self.root, 'CUB_200_2011', 'images.txt')
        labels_path = os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')
        split_path = os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')

        images = pd.read_csv(images_path, sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(labels_path, sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(split_path, sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        """Loads metadata and checks if all files are present."""
        try:
            self._load_metadata()
        except Exception as e:
            print("Metadata load failed:", e)
            return False

        for _, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                return False
        return True

    def _download(self):
        """Downloads and extracts data. Checks if files are valid and exist."""
        if self._check_integrity():
            print('Files already downloaded and verified.')
            return

        print("Downloading dataset...")
        download_url(self.url, self.root, self.filename, self.tgz_md5)
        archive_path = os.path.join(self.root, self.filename)

        print("Extracting files...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.root)
        print("Extraction complete.")

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data)

    def __getitem__(self, idx):
        """Loads an image and the label for it for training and testing."""
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        
        # Adjust from 1-indexed to 0-indexed labels, good for Python.
        target = sample.target - 1
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target