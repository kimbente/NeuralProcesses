import torch
from random import randint
from torch import nn
from torch.distributions.kl import kl_divergence
from utils import (context_target_split, mask_to_np_input, batch_context_target_mask)
from neuralprocess import NeuralProcess2D



class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcess2D instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

       # Check if neural process is for 2D
        self.is_2D = isinstance(self.neural_process, NeuralProcess2D)

        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                # set gradients back to zero after every epoch
                self.optimizer.zero_grad()

                # sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                ### 2D ###
                if self.is_2D:
                    img, _ = data  # data is a tuple (img, label)
                    batch_size = img.size(0)
                    context_mask, target_mask = \
                        batch_context_target_mask(self.neural_process.img_size,
                                                  num_context, num_extra_target,
                                                  batch_size)

                    img = img.to(self.device)
                    context_mask = context_mask.to(self.device)
                    target_mask = target_mask.to(self.device)

                    p_y_pred, q_target, q_context = \
                        self.neural_process(img, context_mask, target_mask)

                    # Calculate y_target as this will be required for loss
                    _, y_target = mask_to_np_input(img, target_mask)

                else:
                    x, y = data
                    # currently just selecting a very small subset of locations
                    x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_extra_target)
                    # pass through NP
                    p_y_pred, q_target, q_context = self.neural_process(x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, average loss: {}".format(epoch, round(epoch_loss / len(data_loader), 3)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # p_y_pred.log_prob(y_target) has shape (batch_size, num_target (may vary), y_dim). 
        # Take mean over batch and sum (log space) over number of targets and dimensions of y
        # sum: size of num_yarget varies
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim = 0).sum()

        # kl_divergence(q_target, q_context) has shape (batch_size, r_dim). 
        # Take mean over batch and sum over r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim = 0).sum()

        # Minimise sum of NLL (negative log likelihood) and kl between distibutions of latent z
        return - log_likelihood + kl