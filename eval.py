import numpy as np

# dice similarity coefficient is used to evaluate the model
def dice_coef(prediction, ground_truth):
    pixels_prediction = list(prediction.getdata())
    pixels_ground_truth = list(ground_truth.getdata())

    pixels_prediction = np.array(pixels_prediction)
    pixels_ground_truth = np.array(pixels_ground_truth)

    # pixel wise multiplication gives 1 only when both are 1
    intersection = np.sum(pixels_prediction * pixels_ground_truth)

    size_pred = pixels_prediction.size
    size_ground_truth = pixels_ground_truth.size

    sum = size_pred + size_ground_truth
    if sum == 0:
        return 1

    similarity_coef = 2 * intersection / sum
    return similarity_coef