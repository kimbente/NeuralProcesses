import numpy as np
import torch
from math import pi
from torch.utils.data import Dataset


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled. (horizontal shift)

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range = (-1., 1.), shift_range = (-.5, .5),
                 num_samples = 1000, num_points = 100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
    

class SineDiscontData(Dataset):
    """
    Dataset of piecewise function 
    for x < bp: 
        f(x) = a * sin(x - b) where a and b are randomly
    for x >= bp:
        f(x) = a * sin(x - b) + c
    The breakpoint bp is fixed.
    The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled. 

    horizontal_shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled. (horizontal shift)

    vertical_shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled. (vertical shift)

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range = (0, 1.), horizontal_shift_range = (-.5, .5),
                 vertical_shift_range = (-1, 1), breakpoint = (0),
                 num_samples = 1000, num_points = 100):
        self.amplitude_range = amplitude_range
        self.horizontal_shift_range = horizontal_shift_range
        self.vertical_shift_range = vertical_shift_range
        self.breakpoint = breakpoint
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = horizontal_shift_range
        c_min, c_max = vertical_shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample horizontal random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Sample vertical random shift
            c = (c_max - c_min) * np.random.rand() + c_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)

            x_low_index, _ = torch.where(x < self.breakpoint)
            x_high_index, _ = torch.where(x >= self.breakpoint)
            x_low = x[x_low_index]
            x_high = x[x_high_index]
            
            y_low = a * torch.sin(x_low - b)
            y_high = (a * torch.sin(x_high - b)) + c
            
            # Concat back together
            y = torch.cat((y_low, y_high), dim = 0)

            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
    
