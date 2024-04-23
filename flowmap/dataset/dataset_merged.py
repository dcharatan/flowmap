import torch
from torch.utils.data import Dataset, IterableDataset


class DatasetMerged(IterableDataset):
    def __init__(self, datasets: list[IterableDataset | Dataset]) -> None:
        self.datasets = datasets

    def __iter__(self):
        remaining = [len(dataset) for dataset in self.datasets]
        iterators = [iter(dataset) for dataset in self.datasets]

        # Yield examples from the union of the datasets in random order.
        while sum(remaining) > 0:
            # Pick a random example (among the pool of all examples).
            chosen_index = torch.randint(0, sum(remaining), tuple())

            # Yield from the corresponding dataset.
            for dataset_index, remaining_from_dataset in enumerate(remaining):
                if chosen_index < remaining_from_dataset:
                    yield next(iterators[dataset_index])
                    remaining[dataset_index] -= 1
                    break
                else:
                    chosen_index -= remaining_from_dataset
            else:
                # If break was never called, something is wrong.
                raise Exception("This should never happen!")

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)
