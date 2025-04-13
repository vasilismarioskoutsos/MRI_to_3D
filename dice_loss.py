import torch

# sizes = (batch, height, width), smooth to prevent division by 0
def dice_loss(prediction, target, smooth=1):
    # sigmoid to convert logits to probabilities
    pred = torch.sigmoid(prediction)
    
    # calc intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice.mean()