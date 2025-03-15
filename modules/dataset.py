import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
import pycountry
from pathlib import Path
from os import listdir
from os.path import isfile, join
import zipfile
from tqdm.notebook import tqdm
from PIL import Image
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
import torch

def download_osv5m_dataset():
    """
    Downloads the OSV5M dataset from Hugging Face if not already present.
    """
    if not os.path.isdir('datasets/osv5m/images'):
        snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", allow_patterns=[
            'images/train/00.zip',
            'images/test/00.zip',
            '*.csv'
            ], repo_type='dataset')

def extract_zip_files(directory="datasets/osv5m"):
    """
    Extracts all zip files in the specified directory.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                    for member in tqdm(zip_ref.infolist(), desc=f'Extracting {os.path.join(root, file)}'):
                        zip_ref.extract(member, root)
                os.remove(os.path.join(root, file))

class ImageCoordinateDataset(Dataset):
    def __init__(self, csv_file, image_dirs, transform=None, max_items=10000):
        self.data = []
        self.files = []
        self.images = []
        self.skipped = 0
        self.csv = csv_file
        self.transform = transform
        self.max_items = max_items

        for image_dir in image_dirs:
            print('Reading', image_dir)

            self.files.extend([f"{image_dir}/{f}" for f in listdir(image_dir) if isfile(join(image_dir, f))])

            print(f'Found {len(self.files)} files.')

            if not os.path.isdir('datasets/osv5m/'):
                os.makedirs('datasets/osv5m/')
            if isfile(f"{csv_file}_filtered.csv"):
                self.df = pd.read_csv(f"{csv_file}_filtered.csv", index_col=False)
            else:
                self.df = pd.concat([chunk for chunk in tqdm(pd.read_csv(self.csv, chunksize=5000, usecols=['id', 'latitude', 'longitude', 'country'], index_col=False), desc='Loading data')])

            print(f'Found {len(self.df)} csv entries.')

            self.df['country'] = self.df['country'].apply(lambda x: pycountry.countries.get(alpha_2=x).name if pycountry.countries.get(alpha_2=x) else x)

            new = pd.DataFrame(columns=['id', 'latitude', 'longitude', 'country'])
            i = 0
            for full_path in tqdm(self.files, total=len(self.files), desc='Processing files'):
                image_name = str(Path(full_path).stem)

                try:
                    row = self.df[self.df['id'] == int(image_name)].iloc[0]
                except:
                    continue
                new.loc[i] = row
                lat = row['latitude']
                lon = row['longitude']

                # Remove mislabelled images (ocean pictures?)
                # if not globe.is_land(float(lat), float(lon)):
                #   self.skipped += 1
                #   continue
                self.data.append(np.array([str(full_path), float(lat), float(lon)]))
                i += 1

            self.df = new
        if self.max_items is not None:
            print(f'Keeping {self.max_items} items')
            self.df = self.df[:self.max_items]
            self.data = self.data[:self.max_items]
            self.files = self.files[:self.max_items]

        self.df.to_csv(f"{csv_file}_filtered.csv", index=False)

        print(f'Dataset ready, {len(self.files)} files.')
        print(f'Skipped {self.skipped} non-land files.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        coordinates = (float(self.data[idx][1]), float(self.data[idx][2]))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(coordinates, dtype=torch.float32), self.data[idx][0]
