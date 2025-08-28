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
            
            kernel_size = random.choice([3, 5, 7])  
            sigma = random.uniform(0.1, 2.0)  
            blured = F.gaussian_blur(main, kernel_size=kernel_size, sigma=sigma)

            return {
                'main': main,
                'blured': blured
            }

        except Exception as e:
            print(f"error while opening image at index {index}, path: {self.paths[index]}")
            raise e


if __name__ == "__main__":
    
    ds = SuperSDataset(
        "E:/Projects/RealProjects/MachineLearningPorjects/FaceDetection/temp/backup/img_align_celeba/valid",
        tf.ToTensor(),
        )
    
    print(ds[10])
