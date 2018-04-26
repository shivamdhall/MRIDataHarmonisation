'''
This file defines functions for generating various different types of visualisations
All the visualisations generated will be stored in a folder that is passed into each 
fucntion.
'''

import numpy as np
import matplotlib.pyplot as plt


def viz_pred(inputs, predictions, labels, sliceNo, title, location):
    # This function takes an input scan, the predicted output scan and the actual output scan.
    # The function then displays a specific axial slice from each scan side-by-side
    # This visualisation is very useful for identifying regions of error in the predicted scans

    maximum = np.max([inputs.max(), predictions.max(), labels.max()])
    fig = plt.figure()
    plt.figure(figsize=(10,10))
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(inputs[:,:,sliceNo,0].T, cmap='gray', origin='lower', vmax = maximum, vmin=0)
    plt.title("Input")
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(predictions[:,:,sliceNo,0].T, cmap='gray', origin='lower', vmax = maximum, vmin=0)
    plt.title("Predicted")
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(labels[:,:,sliceNo,0].T, cmap='gray', origin='lower', vmax = maximum, vmin=0)
    plt.title("Target")
    fig.savefig(location + '/' + title + "_prediction.png")

def viz_diff(predictions, inputs, labels, sliceNo, title, location):
    # This function gnerates a plot of the prediction error
    # (the absolute difference between the predicted and the ground-truth scan for a particular slice)

    maximum = np.max([inputs.max(), predictions.max(), labels.max()])
    difference = np.abs(predictions[:,:,sliceNo,0] - labels[:,:,sliceNo,0])
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2, 3, 2).set_axis_off()
    plt.imshow(np.absolute(difference).T, cmap='nipy_spectral', origin='lower')
    plt.colorbar(shrink=0.2)
    plt.title("Predicted Error")
    plt.subplot(2, 3, 1).set_axis_off()
    plt.imshow(inputs[:,:,sliceNo,0].T, cmap='gray', origin='lower', vmin=0, vmax=maximum)
    plt.title("Input")
    plt.subplot(2, 3, 3).set_axis_off()
    plt.imshow(labels[:,:,sliceNo,0].T, cmap='gray', origin='lower', vmin=0, vmax=maximum)
    plt.title("Target")
    plt.subplot(2, 3, 5).set_axis_off()
    plt.imshow(predictions[:,:,sliceNo,0].T, cmap='gray', origin='lower', vmin=0, vmax=maximum)
    plt.title("Predicted")
    fig.savefig(location + '/' + title + "_prediction_difference.png")


def bland_altman_error(pred, output1, output2, title, location, model):
    # This function generates a plot between the prediction error against the mean of the ground-truth values. 
    # Additonally, a color map is used to plot the percentage error between the ground-truths and the predictions.

    # Do not include background voxel predictions
    if model == "GAN":
        output1_non_background = output1 > 0
        output2_non_background = output2 > 0
        non_background = np.logical_and(output1_non_background, output2_non_background)
        output1 = output1[non_background]
        output2 = output2[non_background]
        pred = pred[non_background]

    output_mean = np.mean([output1, output2], axis=0)
    x_max = np.max(output_mean)
    diff = pred - output_mean  # Difference between predicted and mean
    md = np.mean(diff) # Mean of the difference
    sd = np.std(diff) # Standard deviation of the difference
 
    percentage_error = np.absolute(((diff)/(output_mean))*100)
    percentage_error[percentage_error > 100.0] = 100.0
    fig = plt.figure()
    plt.scatter(output_mean, diff, c=percentage_error, edgecolors='face', cmap='jet')
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.ylabel('prediction error')
    plt.xlabel('Mean output value')
    plt.xlim([0, x_max*1.1]) 
    plt.colorbar()
    plt.title(title)
    fig.savefig(location + '/' + title + "_bland_altman_error.png")



def bland_altman_prediction(pred, output1, output2, title, location, model):
    # This function generates a plot between the predicted voxels against the mean of the ground-truth values. 
    # Additonally, a color map is used to plot the percentage error between the ground-truths and the predictions.

    # Do not include background voxel predictions for GAN model only
    if model == "GAN":
        output1_non_background = output1 > 0
        output2_non_background = output2 > 0
        non_background = np.logical_and(output1_non_background, output2_non_background)
        output1 = output1[non_background]
        output2 = output2[non_background]
        pred = pred[non_background]

    output_mean = np.mean([output1, output2], axis=0)
    x_max = np.max(output_mean)
    diff = pred - output_mean  # Difference between predicted and mean
    md = np.mean(diff) # Mean of the difference
    sd = np.std(diff) # Standard deviation of the difference
 
    percentage_error = np.absolute(((diff)/(output_mean))*100)

    percentage_error[percentage_error > 100.0] = 100.0
    fig = plt.figure()
    plt.scatter(output_mean, pred, c=percentage_error, edgecolors='face', cmap='jet')
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.ylabel('Predicted')
    plt.xlabel('Mean output value')
    plt.xlim([0, x_max*1.1])
    plt.ylim([0, x_max*1.1]) 
    plt.colorbar()
    plt.title(title)
    fig.savefig(location + '/' + title + "_bland_altman_prediction.png")



def plot_losses(training_losses, validation_losses, title, location):
    # This function plots the training losses and validation losses after each epoch

    fig = plt.figure()
    plt.plot(range(1,len(training_losses)+1), training_losses, 'r-', label="Training error")
    plt.plot(range(1,len(validation_losses)+1), validation_losses, 'b-', label="Validation error")
    plt.legend()
    plt.xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel('Loss - (MSE)')
    plt.title(title)
    fig.savefig(location + '/' + title + "_learning_curve.png")
