from torch.utils import data


class BaseDataLoader(data.DataLoader):
    def test_split(self) -> data.DataLoader:
        return NotImplementedError
