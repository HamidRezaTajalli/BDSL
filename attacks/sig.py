import math
from typing import Iterable, Sequence, Optional, Set
import torch
import numpy as np
from torch.utils.data import Dataset

class PoisonedDataset(Dataset):
    """
    Signal-based (SIG) backdoor dataset wrapper.

    Constructor:
        PoisonedDataset(
            args,
            dataset,            # torch.utils.data.Subset
            poisoned_indices,   # Iterable[int] (indices are w.r.t. the *subset*)
            delta,              # amplitude; accepts 30 or 30/255
            f,                  # spatial frequency (integer cycles across width)
            target_label        # int; kept for ASR or bookkeeping
        )

    Behavior:
      - Adds a low-frequency horizontal sinusoid to images whose index is in `poisoned_indices`.
      - Works for grayscale (C=1) and RGB (C=3); auto-detects from first sample or args.dataset.
      - If the sample appears normalized (values outside [0,1]), it will try to denormalize to pixel space,
        add the signal, clamp to [0,1], and then re-apply the same normalization.
      - Training-time labels are kept *clean* (no label poisoning). If you want ASR-style relabeling,
        set `args.relabel_poison_to_target = True`.

    Notes:
      - `poisoned_indices` are assumed to be indices into the provided Subset, *not* the original base dataset.
      - The sinusoid is p[j] = delta * sin(2π f j / W), broadcast over rows & channels.
    """

    _KNOWN_STATS = {
        # means/stds in pixel space (0..1)
        # Common defaults; adjust if your pipeline uses different normalization.
        "mnist":      ([0.1307], [0.3081]),
        "fashionmnist": ([0.2860], [0.3530]),
        "fmnist":     ([0.2860], [0.3530]),
        "cifar10":    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "cifar100":   ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        # GTSRB stats vary by source; many pipelines keep it unnormalized.
        # If your pipeline normalizes, you can add your exact stats here.
        "gtsrb":      (None, None),
    }

    def __init__(
        self,
        args,
        dataset: torch.utils.data.Subset,
        poisoned_indices: Iterable[int],
        delta: float,
        f: int,
        target_label: int,
    ):
        super().__init__()
        if not isinstance(dataset, torch.utils.data.Subset):
            raise TypeError("dataset must be a torch.utils.data.Subset")

        self.args = args
        self.subset = dataset
        self.poisoned_set: Set[int] = set(int(i) for i in poisoned_indices)
        self.target_label = int(target_label)
        self.f = int(f)

        # Accept delta in either [0,1] or [0,255] scale.
        self.delta = float(delta / 255.0) if delta > 1.0 else float(delta)

        # Try to infer dataset name & stats
        ds_name = getattr(args, "dataset", "") or ""
        self.ds_name = ds_name.lower()
        self.ds_mean, self.ds_std = self._infer_stats(self.ds_name)

        # Infer shape from the first sample (handles transformed tensors).
        sample_img, _ = self.subset[0]
        if not torch.is_tensor(sample_img):
            raise TypeError("Expected the subset to return tensors. Apply transforms.ToTensor() upstream.")

        # Normalize shapes to (C,H,W)
        if sample_img.ndim == 2:
            sample_img = sample_img.unsqueeze(0)  # (1,H,W)
        if sample_img.ndim != 3:
            raise ValueError(f"Unexpected image shape {tuple(sample_img.shape)}; expected (C,H,W) or (H,W).")

        C, H, W = sample_img.shape

        # If channels disagree with dataset name, trust the tensor
        if C not in (1, 3):
            # Rare cases (e.g., 4-channel); we will just broadcast pattern to C
            pass

        # Precompute base sinusoid pattern in pixel space, shape (1,H,W)
        self.base_pattern = self._make_pattern(H, W)  # (1,H,W)
        # We will broadcast per-sample to match its channel count on the fly.

        # Whether to relabel poisoned samples to target (useful for ASR evaluation)
        self.relabel_poison = bool(getattr(args, "relabel_poison_to_target", False))

    # ------ core helpers ------

    def _infer_stats(self, name: str):
        for k, (m, s) in self._KNOWN_STATS.items():
            if k in name:
                return (m, s)
        return (None, None)

    @staticmethod
    def _to_float(img: torch.Tensor) -> torch.Tensor:
        # Ensure float32 for math
        if not img.is_floating_point():
            img = img.float()
        return img

    def _denorm_if_needed(self, img: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        If img appears normalized (e.g., negative/ >1), try to denormalize using known stats.
        Returns: (img_denorm, mean_tensor, std_tensor, did_denorm)
        """
        # Heuristic: if in [0,1] already, don't touch.
        if img.min().item() >= 0.0 and img.max().item() <= 1.0:
            return img, None, None, False

        if self.ds_mean is None or self.ds_std is None:
            # No known stats; leave as is (add in current domain).
            return img, None, None, False

        C = img.shape[0]
        mean = torch.tensor(self.ds_mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(self.ds_std, dtype=img.dtype, device=img.device)
        if mean.numel() == 1 and C > 1:
            mean = mean.repeat(C)
            std = std.repeat(C)

        mean = mean.view(C, 1, 1)
        std = std.view(C, 1, 1)

        img_denorm = img * std + mean
        return img_denorm, mean, std, True

    @staticmethod
    def _renorm(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (img - mean) / std

    def _make_pattern(self, H: int, W: int) -> torch.Tensor:
        """
        Create (1,H,W) sinusoid in pixel space: delta * sin(2π f x / W), horizontal stripes.
        """
        xs = torch.arange(W, dtype=torch.float32)
        row = self.delta * torch.sin(2.0 * math.pi * self.f * xs / float(W))  # (W,)
        pat = row.view(1, 1, W).repeat(1, H, 1)  # (1,H,W)
        return pat  # CPU by default; moved to img.device on use

    def _apply_signal(self, img: torch.Tensor) -> torch.Tensor:
        """
        Add the precomputed sinusoid in pixel space, clamped to [0,1].
        Handles denorm -> add -> renorm if necessary.
        """
        img = self._to_float(img)
        original_device = img.device

        # Ensure (C,H,W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        C, H, W = img.shape
        pat = self.base_pattern.to(original_device)
        if pat.shape[1] != H or pat.shape[2] != W:
            # Defensive: resize pattern if spatial dims differ (shouldn't happen in typical setups)
            pat = self._make_pattern(H, W).to(original_device)

        # Broadcast pattern to channels
        pat = pat.repeat(C, 1, 1)  # (C,H,W)

        # Try to go to pixel space if normalized
        img_px, mean, std, did_denorm = self._denorm_if_needed(img)

        # Add & clamp in pixel space
        poisoned_px = torch.clamp(img_px + pat, 0.0, 1.0)

        # Re-normalize if we had denormalized
        if did_denorm:
            poisoned = self._renorm(poisoned_px, mean, std)
        else:
            poisoned = poisoned_px

        return poisoned

    # ------ Dataset interface ------

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]

        # Ensure tensor
        if not torch.is_tensor(img):
            raise TypeError("Expected tensor images from the subset. Make sure transforms.ToTensor() is applied.")

        if idx in self.poisoned_set:
            img = self._apply_signal(img)
            if self.relabel_poison:
                label = self.target_label

        return img, label



def create_sig_poisoned_trainset(args, subset):
    # Find indices of samples that belong to the target class
    target_label_indices = []
    for idx in range(len(subset)):
        _, label = subset[idx]
        if label == args.target_label:
            target_label_indices.append(idx)
    
    # Calculate number of samples to poison from target class
    num_target_samples = len(target_label_indices)
    num_poisoned = int(num_target_samples * args.poisoning_rate)
    
    # Randomly select poisoned indices from target class samples
    if num_poisoned > 0 and num_target_samples > 0:
        poisoned_indices = np.random.choice(target_label_indices, num_poisoned, replace=False)
    else:
        raise ValueError("No target label samples to poison")
    
    # Create poisoned dataset with SIG attack parameters
    poisoned_dataset = PoisonedDataset(
        args=args,
        dataset=subset,
        poisoned_indices=poisoned_indices,
        delta=args.delta,
        f=args.f,
        target_label=args.target_label
    )
    return poisoned_dataset, poisoned_indices


def create_sig_poisoned_testset(args, subset):
    num_samples = len(subset)
    num_poisoned = int(num_samples * args.poisoning_rate)
    poisoned_indices = np.random.choice(num_samples, num_poisoned, replace=False)
    # Create poisoned dataset with SIG attack parameters
    poisoned_dataset = PoisonedDataset(
        args=args,
        dataset=subset,
        poisoned_indices=poisoned_indices,
        delta=args.delta,
        f=args.f,
        target_label=args.target_label
    )
    return poisoned_dataset, poisoned_indices