import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import imgaug as ia
import imgaug.augmenters as iaa

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class SyntheticGenerator(Dataset):
    def __init__(self, object_path, background_path, img_size=416, possible_positions=3, possible_sizes=4, augment=True, multiscale=True, normalized_labels=True):
        self.img_size = img_size
        self.objects_path = object_path
        self.background_path = background_path
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        available_object_classes = [f for f in os.listdir(self.objects_path) if os.path.isdir(os.path.join(self.objects_path, f))]
        self.objects_available = []
        for c in available_object_classes:
            for img in os.listdir(os.path.join(self.objects_path, c)):
                self.objects_available.append({'c': c, 'f': img})
        self.backgrounds = os.listdir(self.background_path)
        # could be parameters
        self.possible_positions = possible_positions
        self.possible_sizes = possible_sizes
        self.augmentation_factor = 2

        self.datasize = len(self.objects_available) * len(self.backgrounds) * self.possible_positions * self.possible_sizes
        self.combinations = []

        object_idx = np.random.randint(0,len(self.objects_available), len(self.objects_available) * len(self.backgrounds))
        bg_idx = np.random.randint(0,len(self.backgrounds), len(self.objects_available) * len(self.backgrounds))
        for e in range(len(self.objects_available) * len(self.backgrounds)):
            o = self.objects_available[object_idx[e]]
            b = self.backgrounds[bg_idx[e]]
            for s in np.random.rand(self.possible_sizes):
                for px in np.random.rand(self.possible_positions):
                    py = np.random.rand()
                    self.combinations.append({
                        'o': o,
                        'b': b,
                        's': 0.5 + (s * 0.5),
                        'p': [px*0.6, py*0.6]
                    })
        assert(self.datasize == len(self.combinations))
        self.seq = iaa.Sequential([
            iaa.ContrastNormalization((0.5, 1))
        ], random_order=True)

    def __getitem__(self, index):
        c = self.combinations[index]

        img_path = os.path.join(self.objects_path, c['o']['c'], c['o']['f'])
        bg_path = os.path.join(self.background_path,c['b'])
        # Extract image as PyTorch tensor
        bg = Image.open(bg_path).convert('RGB')
        obj = Image.open(img_path)
        angle = np.random.rand()*180
        obj = obj.rotate( angle, expand=1 )
        bw, bh = bg.size
        iw, ih = obj.size
        

        bg_dim = bw if bw > bh else bh
        sm_obj_w = iw*c['s']
        sm_obj_h = ih*c['s']
        obj = obj.resize((int(sm_obj_w), int(sm_obj_h)))
        

        r,g,b,a = obj.split()
        rgb_img = Image.merge( 'RGB', (r, g, b))
        
        posx = c['p'][0]
        posy = c['p'][1]
        bg.paste(rgb_img, box=(int(bw*posx), int(bh*posy)), mask=a)
        obj_mid_x = (bw*posx) + sm_obj_w/2
        obj_mid_y = (bh*posy) + sm_obj_h/2

        img = transforms.ToTensor()(bg)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        new_mid_x = (padded_w-bw+0.000001)/2+obj_mid_x
        new_mid_y = (padded_h-bh+0.000001)/2+obj_mid_y

        #calculate label box
        #fields: 'class', 'x', 'y', 'w', 'h'
        boxes = torch.zeros((1,5))
        boxes[:, 0] = float(c['o']['c'])
        boxes[:, 1] = new_mid_x/padded_w
        boxes[:, 2] = new_mid_y/padded_h
        boxes[:, 3] = sm_obj_w/bg_dim*1.06
        boxes[:, 4] = sm_obj_h/bg_dim*1.06

        #fields: 'sample index (set in collate_fn)', 'class', 'x', 'y', 'w', 'h'
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
    
        # Resize
        img = resize(img, self.img_size)
        img_aug = self.seq.augment_image(img)

        return img_path, torch.from_numpy(img_aug), targets
        #return img_path, img, targets

    def __len__(self):
        return self.datasize

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
