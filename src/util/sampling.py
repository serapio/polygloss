import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batches examples of similar length to reduce padding.

    Indices are sorted by length inside fixed-size windows, windows and the
    resulting batches are shuffled each epoch, so padding stays low while the
    example order still changes between epochs.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size_multiplier: int = 20,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.window = max(batch_size, batch_size * bucket_size_multiplier)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        indices.sort(key=self.lengths.__getitem__)
        windows = [
            indices[i : i + self.window]
            for i in range(0, len(indices), self.window)
        ]
        if self.shuffle:
            rng.shuffle(windows)
        batches: list[list[int]] = []
        for window in windows:
            if self.shuffle:
                rng.shuffle(window)
            for i in range(0, len(window), self.batch_size):
                batch = window[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        for i, batch in enumerate(self._batches()):
            if i % self.num_replicas == self.rank:
                yield batch

    def __len__(self) -> int:
        total = len(self._batches())
        return len(range(self.rank, total, self.num_replicas))
