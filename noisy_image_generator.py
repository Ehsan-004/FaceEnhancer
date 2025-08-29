import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch

input_dir = Path("files/original")
output_dir = Path("files/noisy")
output_dir.mkdir(parents=True, exist_ok=True)

transform = T.Compose([
    T.ToTensor()
])

to_pil = T.ToPILImage()

for path in input_dir.rglob("*"):
    img = Image.open(path).convert("RGB")
    main = transform(img)

    kernel_size = random.choice([5, 7, 9])
    sigma = random.uniform(0.4, 6.0)
    blured = F.gaussian_blur(main, kernel_size=kernel_size, sigma=sigma)

    noise_level = random.uniform(0.05, 0.1)
    noise = torch.randn_like(blured) * noise_level
    noisy_blured = torch.clamp(blured + noise, 0.0, 1.0)

    save_path = output_dir / path.name
    to_pil(blured).save(save_path)

print("done")
