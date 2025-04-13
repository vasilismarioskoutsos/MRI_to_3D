import pandas as pd
import h5py
import matplotlib.pyplot as plt

df = pd.read_csv('BraTS20 Training Metadata.csv')
print(df.info())
print(df.isna().sum())

df['slice_path'] = df['slice_path'].replace('/input/brats2020-training-data/', '', regex=True)
df.to_csv('BraTS20_Training_Metadata_Processed.csv', index=False)

def print_h5(h5_path):
    # check a h5 file to see the contents
    h5_path = 'BraTS2020_training_data/content/data/volume_41_slice_0.h5'

    with h5py.File(h5_path, 'r') as h5_file:
        # available keys 
        print("Keys in h5 file:", list(h5_file.keys()))
        
        image = h5_file['image']
        print("Image shape:", image.shape) # 4 channels = T1, T1Gd, T2, T2-FLAIR)
        mask = h5_file['mask'] # contains the tumor regions
        print("Mask shape:", mask.shape)

        # view image and mask
        t1_image = image[:, :, 1]

        plt.figure(figsize=(6,6))
        plt.imshow(t1_image, cmap='gray')
        plt.title('T1-weighted MRI')
        plt.axis('off')
        plt.show()

        mask_channel = mask[:, :, 1]

        plt.figure(figsize=(6,6))
        plt.imshow(mask_channel, cmap='gray')
        plt.title('Mask channel 1')
        plt.axis('off')
        plt.show()