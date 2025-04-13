from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import scipy.ndimage as ndi
import h5py
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]['slice_path'] # get path
        with h5py.File(path, 'r') as h5_file:
            image = h5_file['image']
            mask = h5_file['mask'] # contains the tumor regions
            
            t1_image = image[:, :, 0]
            t1Gd_image = image[:, :, 1]
            t2_image = image[:, :, 2]
            t2flair_image = image[:, :, 3]

            label0 = mask[:, :, 0]
            label1 = mask[:, :, 1]
            label2 = mask[:, :, 2]

            # image that we use from the 4 available
            im = t1_image

            # mask that we use from the 3 labels available
            label = label0

            im = im.astype(np.float32)
            label = label.astype(np.int32)

            # image augmentation
            mean = 0
            sigma = 10
            # make 3x3 coarse grid
            grid = np.zeros((3, 3, 2), dtype=np.float32)
            for i in range(3):
                row = [] 
                for j in range(3):
                    # define gaussian distribution for displacement vector
                    grid[i, j] = np.random.normal(mean, sigma, 2)

            H, W = im.shape
            dx = cv2.resize(grid[:, :, 0], (W, H), interpolation=cv2.INTER_CUBIC)
            dy = cv2.resize(grid[:, :, 1], (W, H), interpolation=cv2.INTER_CUBIC)
            disp = np.stack([dx, dy], axis=-1)

            # grid for the original pixel coords
            grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
            
            # new coordinates 
            new_x = grid_x + disp[:, :, 0]
            new_y = grid_y + disp[:, :, 1]

            # new image using the new coordinates with bicubic interpolation
            new_image = ndi.map_coordinates(im, [new_y, new_x], order=3, mode='reflect')

            # apply the same transform to the label using nearest neighbor interpolation
            new_label = ndi.map_coordinates(label, [new_y, new_x], order=0, mode='reflect')
        return new_image, new_label
