import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

def generate_checkerboard_samples(N, size):
    # Initialize a tensor to hold the N samples of checkerboard patterns
    samples = torch.zeros((N, size, size, 3))  # 3 for RGB channels

    for i in range(N):
        # Generate random colors for the checkerboard
        color1 = torch.rand(3, dtype=torch.float32)  # Color for one set of squares
        color2 = torch.rand(3, dtype=torch.float32)  # Color for the other set of squares

        for x in range(size):
            for y in range(size):
                if (x + y) % 2 == 0:
                    samples[i, x, y] = color1
                else:
                    samples[i, x, y] = color2

    return samples

def visualize(matrix : np.ndarray, title : str = ""):
    if matrix.ndim == 2:  # Grayscale image
        plt.imshow(matrix, cmap='gray', interpolation='none', vmin=0, vmax=1)
    else:  # RGB image
        plt.imshow(matrix, interpolation='none', vmin=0, vmax=1)
    plt.title(title)
    plt.show()


def get_MNIST(grid_size) -> torch.tensor:
    # returns inputs, images of digits

    class FlattenTransform:
        def __call__(self, tensor):
            return tensor.view(-1)

    transform = transforms.Compose([
    transforms.Resize((grid_size, grid_size)),
    transforms.ToTensor(),
    FlattenTransform(),
    ])

    mnist_train = datasets.MNIST('mnist_data', train=True, download=True, transform=transform)

    digit_1_indices = [i for i, label in enumerate(mnist_train.targets) if label == 1]
    temp = [mnist_train[i][0] for i in digit_1_indices]
    digit_1_images = torch.stack([mnist_train[i][0] for i in digit_1_indices])

    return digit_1_images

def get_CIFAR(grid_size) -> torch.tensor:
    # returns inputs, images of cars (class 1 in CIFAR-10)

    transform = transforms.Compose([
        transforms.Resize((grid_size, grid_size)),
        transforms.ToTensor(),
    ])

    cifar_train = datasets.CIFAR10('cifar_data', train=True, download=True, transform=transform)

    class_indices = [i for i, label in enumerate(cifar_train.targets) if label == 1]
    class_images = torch.stack([cifar_train[i][0].permute(1, 2, 0) for i in class_indices])

    return class_images


class MyDataset(Dataset):

    # big part of this function can be vectorized
    def __init__(self, nr_unique_img_without_noise, T, scheduling_fn, data, grid_size=None):
        self.data = data
        self.grid_size = grid_size
        self.nr_unique_img_without_noise = nr_unique_img_without_noise
        self.T = T
        self.scheduling_fn = scheduling_fn 
        self.nr_samples = nr_unique_img_without_noise

        assert grid_size is not None

        if data == 'grid':
            self.nr_channels = 3
            self.__without_noise = generate_checkerboard_samples(self.nr_unique_img_without_noise, self.grid_size)
        elif data == 'mnist':
            self.__without_noise = get_MNIST(self.grid_size)
            self.nr_channels = 1
            assert self.nr_unique_img_without_noise <= len(self.__without_noise)
        elif data == 'cifar10':
            self.__without_noise = get_CIFAR(self.grid_size)
            self.nr_channels = 3
        else:
            assert False

        # for i in range(self.nr_samples):
        #     self.t_arr[i] = torch.randint(1, T + 1, (1,)).item()
        #     self.with_noise[i], self.standard_normal[i] = self.do_fast_noise_skip(self.__without_noise[i//nr_noised_img_per_img_without_noise], self.t_arr[i], scheduling_fn=scheduling_fn)

        self.alpha_t_cum = torch.empty(T+1)
        self.alpha_t_cum[0] = 1
        for cur_t in range(1, T+1):
            beta_t = scheduling_fn(cur_t, self.T)
            self.alpha_t_cum[cur_t] = self.alpha_t_cum[cur_t-1] * (1 - beta_t)

    def __getitem__(self, idx):
        t = torch.randint(1, self.T + 1, (1,))
        
        shape = (self.nr_channels, self.grid_size, self.grid_size)
        with_noise, standard_normal = self.do_fast_noise_skip(self.__without_noise[idx], t, scheduling_fn=self.scheduling_fn)
        
        return (with_noise.view(shape), t), standard_normal.view(shape)
        # is it non optimal to call view here every time?
    
    def get_original_img(self, idx : int):
        assert idx < self.nr_samples
        return self.__without_noise[idx]

    def __len__(self):
        return self.nr_samples

    def do_fast_noise_skip(self, input : torch.tensor, t : torch.tensor, scheduling_fn):
        # might be able to get further optimized

        standard_normal = torch.randn(input.shape)
        noise = standard_normal * torch.sqrt(1- self.alpha_t_cum[t])

        input_with_noise = input * torch.sqrt(self.alpha_t_cum[t]) + noise

        # input_with_noise = torch.clamp(input_with_noise, 0, 1) # do they do this in the original paper?
        return input_with_noise, standard_normal


# def do_noise_skip(input : torch.tensor, t : torch.tensor, beta_t : float):
#     ret = copy.deepcopy(input)

#     for i in range(t):
#         ret = ret * np.sqrt(1 - beta_t) + torch.randn(input.shape) * np.sqrt(beta_t)

#     # ret = torch.clamp(ret, 0, 1) # do they do this in the original paper?
#     return ret