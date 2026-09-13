# ===================================================================================
# MIT License
# 
# Copyright (c) Bernardo (https://github.com/b3r8)
# 
# This file is a modified version of the MVTec Dataloader from:
# https://github.com/b3r8/mvtec-dataloader
# 
# Changes made:
# - Added ground truth (GT) mask extraction and return values during the test phase.
# ===================================================================================

# Ho aggiunto la restituzione della ground truth nei dati test

import os
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from typing import Any, Callable, Optional, Tuple

from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class MVTEC(VisionDataset):
    """`MVTEC <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directories
            ``bottle``, ``cable``, etc., exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        resize (int, optional): Desired output image size (H=W=resize).
        interpolation (int, optional): Interpolation method for downsizing image. If 'resize'
            is not None, a value for interpolation must be provided.
            See https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html
        category (string, optional): bottle, cable, capsule, etc.
    """


    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize: Optional[int] = None,
        interpolation: int = 2,
        category: str = 'carpet',
    ) -> None:
            
        super().__init__(root,
                        transform=transform,
                        target_transform=target_transform)
        
        self.root = os.path.abspath(os.path.expanduser(root))   # !! MODIFICHE QUI !!
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.interpolation = interpolation
        self.category = category

        self.data = []
        self.targets = []

        self.masks = []     # !! MODIFICHE QUI !!
        
        if self.train:
            # load images for training
            cwd = os.getcwd()
            trainFolder = os.path.join(self.root, self.category, 'train/good/')
            os.chdir(trainFolder)

            for file in os.scandir():
                img = mpimg.imread(file.name)
                img = img*255
                img = img.astype(np.uint8)
                self.data.append(img)
                
                # label 1 = 'good' image
                self.targets.append(1)
                
            os.chdir(cwd)      
        else:
            # load images for testing
            cwd = os.getcwd()
            testFolder = os.path.join(self.root, self.category, 'test/')

            gtFolder = os.path.join(self.root, self.category, 'ground_truth/')  # !! MODIFICHE QUI !!

            os.chdir(testFolder)

            subfolders = sorted([subfolder.name for subfolder in os.scandir() if subfolder.is_dir()])   # !! MODIFICHE QUI !!

            cwsd = os.getcwd()
            
            # for every subfolder in test folder
            for subfolder in subfolders:
                # label 0 = 'defective' image
                label = 0
                if subfolder == 'good':
                    label = 1

                os.chdir(subfolder)
                #filenames = [file.name for file in os.scandir()]
                #for file in filenames:
                for file in sorted(os.scandir(), key=lambda e: e.name): # !! MODIFICHE QUI !!
                    img = mpimg.imread(file.name)
                    img = img*255
                    img = img.astype(np.uint8)
                    self.data.append(img)
                    self.targets.append(label)

                    if label == 1:
                        # le immagini 'good' non hanno difetti, quindi creiamo una maschera tutta nera (zeri)
                        h, w = img.shape[0], img.shape[1]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        self.masks.append(mask)
                    else:
                        # le immagini difettose hanno la maschera in ground_truth/nome_difetto/nome_file_mask.png
                        base_name, _ = os.path.splitext(file.name)
                        mask_name = f"{base_name}_mask.png"
                        mask_path = os.path.join(gtFolder, subfolder, mask_name)
                        
                        if os.path.exists(mask_path):
                            mask = mpimg.imread(mask_path)
                            if mask.max() <= 1.0:
                                mask = mask * 255
                            mask = mask.astype(np.uint8)
                            # se la maschera è salvata a 3 canali (RGB), prendiamo solo il primo canale (monocromatico)
                            if len(mask.shape) == 3:
                                mask = mask[:, :, 0]
                            self.masks.append(mask)
                        else:
                            # fallback di sicurezza in caso di file mancanti ma abbastanza inutile
                            h, w = img.shape[0], img.shape[1]
                            mask = np.zeros((h, w), dtype=np.uint8)
                            self.masks.append(mask)
                    
                os.chdir(cwsd)
                
            os.chdir(cwd)

        # data (images) is a numpy array,
        # targets (labels) is a list
        self.data = np.array(self.data)

        # print original data shape to screen
        print('original data shape: (N, H, W, C)', self.data.shape)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is 0 for 'defective' images
                and 1 for 'good' images
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # if resizing image
        # See: https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
        if self.resize:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # !! MODIFICHE QUI !!
        if not self.train:
            mask = self.masks[index]
            mask = Image.fromarray(mask)
            if self.resize:
                # NEAREST interpolation per evitare bordi sfumati/grigi sulla maschera binaria
                resizeMask = transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.NEAREST)
                mask = resizeMask(mask)
            
            mask_tensor = transforms.ToTensor()(mask)
            return img, mask_tensor, target
        # !! FINE MODIFICHE !!
        
        return img, target

    def __len__(self) -> int:
        """
        Args:
            None
        Returns:
            int: length of data
        """
        return len(self.data)
