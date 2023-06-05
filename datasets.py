import numpy as np
import torch
import gpytorch
from math import pi
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


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
        f(x) = a * sin(x - b)
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
    def __init__(self, amplitude_range = (0.5, 1.), horizontal_shift_range = (-.5, .5),
                 vertical_shift_range = (0, 1), breakpoint = (0),
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
    

def mnist(batch_size = 16, size = 28):
    """MNIST dataloader. 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    Returns
    -------
    dataloader objects
    """
    # Compose various transformations together
    combined_transforms = transforms.Compose([
        # Resize the input image to the given size.
        transforms.Resize(28),
        # Convert a PIL Image or ndarray to tensor and scale the values accordingly.
        transforms.ToTensor(),
    ])

    train_data = datasets.MNIST(root = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', train = True, 
                                download = True, transform = combined_transforms)

    test_data = datasets.MNIST(root = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', train = False, 
                               download = True, transform = combined_transforms)

    # Store into DataLoader object
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader


#### GP ####

class LinearGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, init_linearmean_weight):
        super(LinearGPModel, self).__init__(train_x, train_y, likelihood)

        self._init_linearmean_weight = init_linearmean_weight

        ### LINEAR MEAN ###
        self.mean_module = gpytorch.means.LinearMean(input_size = 1)
        self.mean_module.initialize(weights = self._init_linearmean_weight)
        # Will not be updated during training
        self.mean_module.weights.requires_grad = False

        ### KERNEL ###
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class RBFGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, init_lengthscale):
        super(RBFGPModel, self).__init__(train_x, train_y, likelihood)

        self._init_lengthscale = init_lengthscale

        ### LINEAR MEAN ###
        self.mean_module = gpytorch.means.ConstantMean(input_size = 1)

        ### KERNEL ###
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = self._init_lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# inherits Dataset from torch.utils.data
class GP_coin(Dataset):
    """
    Dataset created from linear kernel and rbf kernel.

    Parameters
    ----------
    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, num_samples = 1000, num_points = 100):
        # Num
        self.num_samples = num_samples
        self.num_points = num_points

        # Dim
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        ### MODELS ###
        # LINEAR
        # Load state dict
        linear_state_dict = torch.load('/experiments_1D/data/GP_models/model_state_linear_model.pth')
        # Initialise model
        train_x = torch.tensor([])
        train_y = torch.tensor([])
        init_linearmean_weight = torch.tensor(1, dtype = torch.int64)   
        # Likelihood
        linear_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = torch.tensor([0.2]), learn_additional_noise = False)
        linear_model = LinearGPModel(train_x, train_y, linear_likelihood, init_linearmean_weight)
        # Load state dict into model
        linear_model.load_state_dict(linear_state_dict)
        # Assign
        self._linear_model = linear_model

        # RBF
        # Load state dict
        rbf_state_dict = torch.load('/experiments_1D/data/GP_models/model_state_rbf_model.pth')
        # Intialise model
        train_x = torch.tensor([-3, -2, 0, 2, 3])
        train_y = torch.tensor([2.5, 3.5, 0, -3.5, -2.5])
        init_lengthscale = torch.tensor(1, dtype = torch.int64)
        # Likelihood
        rbf_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = torch.tensor([.1]), learn_additional_noise = True)
        rbf_model = RBFGPModel(train_x, train_y, rbf_likelihood, init_lengthscale)
        # Load state dict into model
        rbf_model.load_state_dict(rbf_state_dict)
        self._rbf_model = rbf_model

        # Generate data
        self.data = []

        x = torch.linspace(-pi, pi, num_points).unsqueeze(1)

        for i in range(num_samples):
            coin = torch.rand(1)

            if coin > 0.5:
                linear_f_preds = linear_model(x)
                y = linear_f_preds.sample()

            else:
                rbf_f_preds = rbf_model(x)
                y = rbf_f_preds.sample()
            
            self.data.append((x, y))
            
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples