import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from IPython.display import clear_output as clc
from IPython.display import display

mae = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()
mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"


class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def Padding(data, lag):
    '''
    Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
    '''
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out


def multiplot(yts, plot, titles = None, fontsize = None, figsize = None, vertical = False, axis = False, save = False, name = "multiplot", add_points = None):
    """
    Multi plot of different snapshots
    Input: list of snapshots, related plot function, plot options, save option and save path
    """
    
    plt.figure(figsize = figsize)
    for i in range(len(yts)):
        if vertical:
            plt.subplot(len(yts), 1, i+1)
        else:
            plt.subplot(1, len(yts), i+1)
        plot_fn = plot[i] if isinstance(plot, (list, tuple)) else plot
        plot_fn(yts[i])
        plt.scatter(add_points[:, 0], add_points[:, 1], color='red', s=10) if add_points is not None else None
        if titles is not None:
            plt.title(titles[i], fontsize = fontsize)
        if not axis:
            plt.axis('off')
    
    if save:
        plt.savefig(name.replace(".png", "") + ".png", transparent = True, bbox_inches='tight')
        print(f"Plot saved to {name.replace('.png', '') + '.png'}")


def trajectory(yt, plot, title = None, fontsize = None, figsize = None, axis = False, save = False, name = 'gif', add_points = None):
    """
    Trajectory gif
    Input: trajectory with dimension (sequence length, data shape), related plot function for a snapshot, plot options, save option and save path
    """

    arrays = []
        
    for i in range(yt.shape[0]):
        plt.figure(figsize = figsize)
        plot(yt[i])
        plt.title(title, fontsize = fontsize)
        if not axis:
            plt.axis('off')
        plt.scatter(add_points[:, 0], add_points[:, 1], color='red', s=10) if add_points is not None else None
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)
        print(f"Trajectory gif saved to {name.replace('.gif', '') + '.gif'}")
        

def trajectories(yts, plot, titles=None, fontsize=None, figsize=None, vertical=False, axis=False, save=False, name='gif', add_points=None):
    """
    Gif of different trajectories optimized for headless HPC environments.
    Input: list of trajectories with dimensions (sequence length, data shape), 
           plot function for a snapshot, plot options, save option and save path
    """
    arrays = []
    
    # Ensure the name is a string so .replace() and string concatenation work
    name_str = str(name)

    # Loop through the time steps
    for i in range(yts[0].shape[0]):
        fig = plt.figure(figsize=figsize)
        
        # Loop through the different trajectories (e.g., Ground Truth, Prediction)
        for j in range(len(yts)):
            if vertical:
                plt.subplot(len(yts), 1, j+1)
            else:
                plt.subplot(1, len(yts), j+1)
                
            plot(yts[j][i])
            
            if titles is not None:
                plt.title(titles[j], fontsize=fontsize)
            if not axis:
                plt.axis('off')

        if save:
            # CRITICAL FIX: Force Matplotlib to render the image in the background
            fig.canvas.draw()
            # Safely extract the RGBA buffer
            image = np.array(fig.canvas.buffer_rgba())
            arrays.append(image)
            
        # Close the figure to prevent massive RAM leaks!
        plt.close(fig)

    if save:
        # Format the save path correctly
        save_path = name_str.replace(".gif", "") + ".gif"
        imageio.mimsave(save_path, arrays)
        print(f"Trajectory gif saved to {save_path}")