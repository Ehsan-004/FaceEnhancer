import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import random


class UFaceDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.paths = [p for p in Path(path).rglob("*.jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            img = Image.open(self.paths[index])
            main = self.transform(img)

            kernel_size = random.choice([5, 7, 9])
            sigma = random.uniform(0.4, 6.0)
            blured = F.gaussian_blur(main, kernel_size=kernel_size, sigma=sigma)

            # Make noise level random
            noise_level = random.uniform(0.05, 0.1)
            noise = torch.randn_like(blured) * noise_level
            noisy_blured = torch.clamp(blured + noise, 0.0, 1.0)

            return {
                'main': main,
                'noisy': noisy_blured
            }

        except Exception as e:
            print(f"error while opening image at index {index}, path: {self.paths[index]}")
            raise e


if __name__ == "__main__":
    
    ds = UFaceDataset(
        "E:/Projects/RealProjects/MachineLearningPorjects/FaceDetection/temp/backup/img_align_celeba/valid",
        tf.ToTensor(),
        )
    
    print(ds[10])
