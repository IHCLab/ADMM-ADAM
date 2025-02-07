import numpy as np
import torch, os
import random as rn
from torch.utils import data
from torch.utils.data import DataLoader
import torch.multiprocessing
from scipy.io import loadmat

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, X, img_size=256, root='', mode='Train'):
        super(dataset_h5, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))
        self.mode=mode
        self.img_size=img_size
    
        
    def __getitem__(self, index):
        
        fn = os.path.join(self.root, self.fns[index])

        x=loadmat(fn)
        x=x[list(x.keys())[-1]]

        x = x.astype(np.float32)
        xmin = np.min(x)
        xmax = np.max(x)
        x = (x-xmin) / (xmax-xmin)

        return x, fn #image, name

    def __len__(self):
        return self.n_images