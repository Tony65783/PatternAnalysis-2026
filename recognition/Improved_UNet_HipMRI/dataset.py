
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


DATA_ROOT = Path('/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')


class HipMRIDataset(Dataset):


    def __init__(self, split='train', root=DATA_ROOT):
        if split not in ('train', 'validate', 'test'):
            raise ValueError('split must be train, validate, or test')
        root = Path(root)
        self.image_dir = root / f'keras_slices_{split}'
        self.mask_dir = root / f'keras_slices_seg_{split}'
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f'Missing split directories: {self.image_dir}, {self.mask_dir}')

        self.pairs = []
        for image_path in sorted(self.image_dir.glob('case_*.nii.gz')):
            mask_name = 'seg_' + image_path.name[len('case_'):]
            mask_path = self.mask_dir / mask_name
            if not mask_path.is_file():
                raise FileNotFoundError(f'Missing mask for {image_path.name}: {mask_path}')
            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise RuntimeError(f'No MRI files found in {self.image_dir}')
        if len(self.pairs) != len(list(self.mask_dir.glob('seg_*.nii.gz'))):
            raise RuntimeError('Image and mask counts differ; check filenames')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.asarray(nib.load(str(image_path)).get_fdata(dtype=np.float32)).squeeze()
        mask = np.asarray(nib.load(str(mask_path)).get_fdata(dtype=np.float32)).squeeze()

        if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
            raise ValueError(f'Invalid image/mask shapes: {image_path.name}: {image.shape}, {mask.shape}')
        if not np.isfinite(image).all() or not np.isfinite(mask).all():
            raise ValueError(f'Non-finite values in {image_path.name}')
        if not np.all(mask == np.rint(mask)):
            raise ValueError(f'Non-integer labels in {mask_path.name}')

        # Per-slice min-max scaling; safe for constant-intensity slices.
        lo, hi = float(image.min()), float(image.max())
        image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor


if __name__ == '__main__':
    for split in ('train', 'validate', 'test'):
        dataset = HipMRIDataset(split)
        image, mask = dataset[0]
        print(f'{split}: {len(dataset)} slices; image={tuple(image.shape)} '
              f'mask={tuple(mask.shape)} labels={torch.unique(mask).tolist()} '
              f'intensity=[{image.min().item():.3f}, {image.max().item():.3f}]')
