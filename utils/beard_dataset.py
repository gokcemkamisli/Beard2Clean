import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BeardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = {}
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.png'):
                parts = filename.split('_')
                if len(parts) != 3:
                    continue
                pair_id = parts[1]
                label = parts[2].split('.')[0]
                if pair_id not in self.pairs:
                    self.pairs[pair_id] = {}
                self.pairs[pair_id][label] = filename
        self.pair_keys = [key for key, pair in self.pairs.items() if 'clean' in pair and 'beard' in pair]
        self.pair_keys.sort()

    def __len__(self):
        return len(self.pair_keys)

    def __getitem__(self, idx):
        key = self.pair_keys[idx]
        pair = self.pairs[key]
        beard_path = os.path.join(self.data_dir, pair['beard'])
        clean_path = os.path.join(self.data_dir, pair['clean'])
        beard_img = Image.open(beard_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        if self.transform:
            beard_img = self.transform(beard_img)
            clean_img = self.transform(clean_img)
        return {'input': beard_img, 'target': clean_img}

# transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
