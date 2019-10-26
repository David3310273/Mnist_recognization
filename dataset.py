from torchvision.datasets import ImageFolder
from torchvision import get_image_backend
from PIL import Image
import os

def _default_loader(path):
    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            """
            注意此处要转成灰度图
            """
            return img.convert('L')

    def accimage_loader(path):
        import accimage
        try:
            return accimage.Image(path);
        except IOError:
            return pil_loader(path);

    if get_image_backend() == 'accimage':
        return accimage_loader(path);
    else:
        return pil_loader(path);

class ImageDataSet(ImageFolder):
    """docstring for ImageLoader"""
    def __init__(self, root, extensions=('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'), transform=None, target_transform=None):
        # super().__init__(root, loader=self._default_loader);
        self.transform = transform;
        self.root = root;
        self.target_transform = target_transform;

        self.loader = _default_loader;
        self.extensions = extensions

        classes, class_to_idx = self._find_classes(self.root);

        samples = self._make_dataset(self.root, class_to_idx, extensions);

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.classes = classes;
        self.class_to_idx = class_to_idx;
        self.samples = samples;
        self.targets = [s[1] for s in samples];
        self.imgs = self.samples;

    def _make_dataset(self, directory, class_to_idx, extensions=None):
        images = [];
        directory = os.path.expanduser(directory);

        for key, val in class_to_idx.items():
            path = os.path.join(directory, key);
            images.append((path, val));
        return images;

    def _find_classes(self, dir):
        class_to_idx = {}
        for filename in os.listdir(self.root):
            name, ext = filename.split(".");
            if ext in self.extensions:
                class_to_idx[filename] = int(name.split("_")[1]);
        return None, class_to_idx;

    
