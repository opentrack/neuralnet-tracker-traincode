from torch.utils.data import IterableDataset, get_worker_info
import numpy as np


class InfinitelyRepeatingInterleavedDatasets(IterableDataset):
    def __init__(self, stop_after, datasets, probs):
        super(InfinitelyRepeatingInterleavedDatasets, self).__init__()
        self.stop_after = stop_after
        self.datasets = datasets
        self.probs = (np.array(probs) / np.sum(probs))
        assert len(datasets) == len(probs)
        
    def __iter__(self):
        worker_info = get_worker_info()
        stop_after = self.stop_after
        if worker_info is not None:
            stop_after //= worker_info.num_workers

        iters = [
            iter(ds) for ds in self.datasets
        ]
        for _ in range(stop_after):
            i, = np.random.choice(len(iters), size=1, p=self.probs)
            try:
                item = next(iters[i])
            except StopIteration:
                iters[i] = iter(self.datasets[i])
                item = next(iters[i])
            yield item
    
    def __len__(self):
        return self.stop_after