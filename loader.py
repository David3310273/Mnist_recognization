from torch.utils.data import DataLoader

class ImageLoader(DataLoader):
    """docstring for ImageLoader"""

    def __init__(self, train_dataset, batch_size=100, num_workers=0, shuffle=True):
        super(ImageLoader, self).__init__(train_dataset, batch_size=batch_size, num_workers=num_workers);
