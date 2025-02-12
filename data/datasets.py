import os
import tarfile
import random
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
import torch

class Cub2011(torch.utils.data.Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

    def _load_metadata(self):
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
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        # Adjust from 1-indexed to 0-indexed labels
        target = sample.target - 1
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target