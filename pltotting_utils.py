""" This file includes Python code licensed under the MIT License, 
    Copyright (c) 2018 Bjarte Mehus Sunde. """

import numpy as np
import matplotlib.pyplot as plt


""" Function used to set some style configurations. """
def set_default(figsize=(10, 10), dpi=100):
    # set the 'Solarize_Light2' default style
    plt.style.use(['Solarize_Light2'])
    # set the background color of the axes to dark gray
    plt.rcParams['axes.facecolor'] = '#2C2C2C'
    # set the background color of the figure to dark gray
    plt.rcParams['figure.facecolor'] = '#2C2C2C'
    # set the text color on the axes to white
    plt.rcParams['text.color'] = 'white'
    # set the color of the lines to white
    plt.rcParams['lines.color'] = 'white'
    # set the figure size and DPI
    plt.rc('figure', figsize=figsize, dpi=dpi)

    # plt.style.use(['Solarize_Light2']) #, 'dark_background'
    # plt.rc('axes', facecolor='k')
    # plt.rc('figure', facecolor='k')
    # plt.rc('figure', figsize=figsize, dpi=dpi)

""" Function used to visualize the training and validation loss. 
    The function also adds a vertical dashed line to the plot to 
    indicate the early stopping checkpoint. """
def plot_losses(train_loss, valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 10))
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('num_epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(train_loss + valid_loss)+0.1)  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig("./loss_plot.png", bbox_inches="tight")

    return fig

def plot_confusion_matrix(tn, fp, fn, tp):
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    fig = plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)

    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])

    for i in range(2):
        for j in range(2):
            value = confusion_matrix[i, j]
            color = 'w' if value > confusion_matrix.max() / 2 else 'k'
            plt.text(j, i, str(value), ha="center", va="center", color=color)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.colorbar()

    return fig
