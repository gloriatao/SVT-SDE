import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import random, numbers

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = (int(size[0]), int(size[1]))
        else:
            self.size = size

    def __call__(self, img, seed=None):
        return TF.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, seed=None):
        if seed == None:
            seed = random.randint(0, 1e5)

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(seed)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print('crop:', seed, x1, y1)
        return img.crop((x1, y1, x1 + tw, y1 + th))

class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img, seed=None):
        if seed == None:
            seed = random.randint(0, 1e5)
        random.seed(seed)
        prob = random.random()
        # print('flip:', seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img, seed=None):
        if seed == None:
            seed = random.randint(0, 1e5)
        random.seed(seed)
        angle = random.randint(-self.degrees, self.degrees)
        # print('angle:', seed, angle)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img, seed=None):
        if seed == None:
            seed = random.randint(0, 1e5)
        random.seed(seed)
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)
        # print('color:',seed,brightness_factor,contrast_factor,saturation_factor,hue_factor)

        return img_

class Compose(object):
    # compose with fixed random seed
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seed=None):
        for t in self.transforms:
            img = t(img, seed)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    def __call__(self, pic, seed):
        return TF.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor, seed):
        return TF.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)