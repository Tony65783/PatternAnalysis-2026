
import torch
from modules import UNet, ImprovedUNet

device = torch.device("cuda")

models = [
    ("Baseline U-Net", UNet(in_channels=1, num_classes=6)),
    ("Improved U-Net", ImprovedUNet(in_channels=1, num_classes=6)),
]

for name, model in models:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = model.to(device).eval()
    image = torch.randn(1, 1, 256, 256, device=device)

    with torch.inference_mode():
        for _ in range(10):
            output = model(image)
        torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"{name}: {peak_mb:.2f} MiB peak GPU memory")

    del model, image, output
    torch.cuda.empty_cache()

