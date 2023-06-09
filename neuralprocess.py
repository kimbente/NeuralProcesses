import torch
# import three coders from coders.py
from coders import rEncoder, zEncoder, Decoder
# import helper function from utils.py
from utils import mask_to_np_input

from torch import nn
from torch.distributions import Normal


class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layers in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = rEncoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = zEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        """
        Aggregates representations r_i for all (x_i, y_i) pairs into a single
        representation r_agg of dimensionality r_dim.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        
        Returns
        -------
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        # Mean over all points
        return torch.mean(r_i, dim = 1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z. (Encoding)

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors by combining batch_size and num_points into the first dimensions, as encoder expects "two" dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        # contiguous() to address memory issues
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)

        ### rEncoder ###
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)

        ### zEndcoder ###
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target = None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size() # batch_size and num_context must match above 

        # Set neuralprocess.training = True or neuralprocess.training = False
        if self.training:
            # Encode target and context (target needs to be encoded to
            # calculate kl term)
            # During training pass target pairs through encoder aswell
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)

            # Sample from encoded distribution using reparameterization trick
            # distribution object
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            # sample from z distribution: rsample: sampling using reparameterization trick. Keeps comp. graph alive and thus allows backprop.
            # During training: Sample from q_target since this is the suberset
            # PREVIOUSLY
            # z_sample = q_target.rsample()
            z_sample = q_context.rsample()

            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            # Predictive distribution
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            # For KL? Pred based on q_target
            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred
        
### Spatial model ###

class NeuralProcess2D(nn.Module):
    """
    Wraps regular Neural Process for image processing.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 28, 28) or (3, 32, 32)

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, img_size, r_dim, z_dim, h_dim):
        super(NeuralProcess2D, self).__init__()
        self.img_size = img_size
        self.num_channels, self.height, self.width = img_size
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.neural_process = NeuralProcess(x_dim = 2, y_dim = self.num_channels,
                                            r_dim = r_dim, z_dim = z_dim,
                                            h_dim = h_dim)

    def forward(self, img, context_mask, target_mask):
        """
        Given an image and masks of context and target points, returns a
        distribution over pixel intensities at the target points.

        Parameters
        ----------
        img : torch.Tensor
            Shape (batch_size, channels, height, width)

        context_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as context.

        target_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as target.
        """
        x_context, y_context = mask_to_np_input(img, context_mask)
        x_target, y_target = mask_to_np_input(img, target_mask)
        return self.neural_process(x_context, y_context, x_target, y_target)