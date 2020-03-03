from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def split_validation(self) -> DataLoader:
        return NotImplementedError
