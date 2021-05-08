import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

from config import BATCH_SIZE

NORMAL = 0
BACTERIAL_PNEUMONIA = 1
VIRAL_PNEUMONIA = 2


class CustomImageDataset(Dataset):
    def __init__(self,  data_dir, transform):
        self.classes = ('NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA')
        self.targets = (0, 1, 2)
        self.batch_size = BATCH_SIZE
        self.transform = transform
        self.data_dir = data_dir

        data = []
        for group in os.listdir(data_dir):
            for img in os.listdir(data_dir + "\\" + group):
                if group == "NORMAL":
                    data.append({'path': group + "/" + img, 'label': NORMAL})
                elif group == "PNEUMONIA":
                    if "bacteria" in img:
                        data.append({'path': group + "/" + img, 'label': BACTERIAL_PNEUMONIA})
                    elif "virus" in img:
                        data.append({'path': group + "/" + img, 'label': VIRAL_PNEUMONIA})

        self.image_dataset = data

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image_data = self.image_dataset[idx]
        img_path = os.path.join(self.data_dir, image_data['path'])
        image = read_image(img_path, ImageReadMode.RGB).float()
        image = self.transform(image)

        return image.float(), image_data['label']
