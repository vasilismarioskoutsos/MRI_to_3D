import pandas as pd
import numpy as np                
import torch
from train_loader import TrainDataset  
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR 
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from model import UNet
from dice_loss import dice_coef
import time
import os
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import tetgen
from PIL import Image   


df = pd.read_csv('BraTS20_Training_Metadata_Processed.csv')

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

learning_rate = 1e-3

model = UNet(in_channels=1, out_channels=1)  
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)
epoch_num = 50
criterion = nn.BCEWithLogitsLoss()

# dataset and dataLoader 
train_dataset = TrainDataset(df_train)  
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

#  test dataset and test loader
test_dataset = TrainDataset(df_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

run_num = 1
epoch_time = []
all_batch_times = []  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(epoch_num):
    start_epoch = time.time()
    model.train()  
    epoch_loss = 0.0
    batch_time = []
    
    for batch in train_loader:  
        start_batch = time.time()
        
        image, label = batch  
        image = image.to(device)     
        label = label.to(device)     
        prediction = model(image)

        probabilities = torch.sigmoid(prediction)
        mask = (probabilities > 0.5).float()
        
        loss = criterion(prediction, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        end_batch = time.time()
        batch_duration = end_batch - start_batch
        batch_time.append(batch_duration)
        all_batch_times.append(batch_duration)
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epoch_num}, Loss: {avg_loss:.4f}")
    
    # model eval
    model.eval() 
    dice_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for image, gt in test_loader:
            image = image.to(device)
            gt = gt.to(device)
            logits = model(image)
            probabilities = torch.sigmoid(logits)
            mask = (probabilities > 0.5).float()
            
            # Convert tensors to numpy arrays and then scale to [0,255] for PIL conversion.
            mask_np = (mask.cpu().numpy().squeeze() * 255).astype(np.uint8)
            gt_np = (gt.cpu().numpy().squeeze() * 255).astype(np.uint8)
            pred_img = Image.fromarray(mask_np)
            gt_img = Image.fromarray(gt_np)
            
            dice_sum += dice_coef(pred_img, gt_img)
            num_batches += 1
    average_dice = dice_sum / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch+1}/{epoch_num}, Dice Coefficient: {average_dice:.4f}")
    
    # save checkpoints
    os.makedirs(f"run_{run_num}", exist_ok=True)
    if (epoch + 1) % 10 == 0:
        path = os.path.join(f"run_{run_num}", f"unet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), path)
    
    end_epoch = time.time()
    epoch_time.append(end_epoch - start_epoch)

print(f"Average epoch time: {sum(epoch_time) / len(epoch_time)}")
print(f"Average batch time: {sum(all_batch_times) / len(all_batch_times)}")

# test segmentation

model.eval()
predicted_slices = []  # collect each 2D binary mask

with torch.no_grad():
    for image, _ in test_loader:
        image = image.to(device)
        logits = model(image)  
        probabilities = torch.sigmoid(logits)
        mask = (probabilities > 0.5).float()  # binary mask

        mask_np = mask.cpu().detach().numpy().squeeze()
        predicted_slices.append(mask_np)

# stack the individual 2d arrays into 3D volume
volume = np.stack(predicted_slices, axis=0)

# 3D reconstruction 
# 3D binary volume to surface mesh
verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

# ivsualize 3d surface mesh 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.7)
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.tight_layout()
plt.show()

# convert polyhedral to tetrahedral model
pv.set_plot_theme('document')
tet = tetgen.TetGen(verts, faces)
tet.tetrahedralize(order=1, mindihedral =20, minratio = 1.5)
grid = tet.grid
grid.plot(show_edges=True)