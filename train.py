import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from logger import TgaLogger
from model import Generator, Discriminator, weights_init
from utils.classify_type import ClassifierType


class Trainner:
    def __init__(self):
        self.__logger = TgaLogger('trainer').get_logger()
        self.__model_params = {
            'nc': 1,  # Number of channels in the training images. For color images this is 3
            'nz': 100,  # Size of z latent vector (i.e. size of generator input)
            'ngf': 32,  # Size of feature maps in generator
            'ndf': 32  # Size of feature maps in discriminator
        }
        self.__workers = 8  # Number of workers for dataloader
        self.__batch_size = 64  # Batch size during training
        self.__image_size = 32  # Spatial size of training images. Images will be resized to this using a transformer.
        # Number of training epochs
        self.__num_epochs = 10
        self.__lr = 0.0002  # Learning rate for optimizers
        self.__beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers
        self.__ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
        # Decide which device we want to run on
        self.__device = torch.device("cuda:0" if (torch.cuda.is_available() and self.__ngpu > 0) else "cpu")

        self.__init_random()
        self.__init_model_loss_opt()

        # Lists to keep track of progress
        self.__img_list = []
        self.__G_losses = []
        self.__D_losses = []

        self.__log_params()

    def __log_params(self):
        self.__logger.debug('#' * 10 + ' Params ' + '#' * 10)
        self.__logger.info(f'workers: {self.__workers}'),
        self.__logger.info(f'batch size: {self.__batch_size}')
        self.__logger.info(f'image size: {self.__image_size}')
        self.__logger.info(f'epochs: {self.__num_epochs}')
        self.__logger.info(f'learning rate: {self.__lr}')
        self.__logger.info(f'beta1: {self.__beta1}')
        self.__logger.info(f'n_gpu: {self.__ngpu}')
        self.__logger.info(f'device: {self.__device}')
        self.__logger.info(f'model_params: {self.__model_params}')

    def __init_random(self):
        self.__logger.debug('Init random ...')
        self.__manualSeed = 999  # Set random seed for reproducibility
        # manualSeed = random.randint(1, 10000) # use if you want new results
        self.__logger.info(f'manualSeed: {self.__manualSeed}')
        random.seed(self.__manualSeed)
        torch.manual_seed(self.__manualSeed)
        torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    def __init_dataset(self, classify_type: ClassifierType):
        self.__logger.debug('Init dataset ...')
        self.__data_root = os.path.realpath('./resources/img/rfc/test')  # Root directory for dataset
        self.__logger.info(f'Data path: {self.__data_root}')
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        self.__logger.info('Loading data ...')
        self.__dataset = dset.ImageFolder(root=self.__data_root, transform=transforms.Compose([
            transforms.Grayscale(),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]))
        self.__logger.info('Loading data DONE!')
        self.__logger.info(f'Data :{self.__dataset.class_to_idx}')
        # Create the dataloader
        self.__dataloader = torch.utils.data.DataLoader(self.__dataset, batch_size=self.__batch_size, shuffle=True,
                                                        num_workers=self.__workers)

    def __plot_training_imgs(self):
        # Plot some training images
        real_batch = next(iter(self.__dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.__device)[:64], padding=2, normalize=True).cpu(),
                                (1, 2, 0)))
        plt.show()

    def __init_model_loss_opt(self):
        self.__logger.debug('Init model, loss and opt ...')

        # Create the generator
        self.__generator = Generator(self.__model_params).to(self.__device)
        # Handle multi-GPU if desired
        if (self.__device.type == 'cuda') and (self.__ngpu > 1):
            self.__generator = nn.DataParallel(self.__generator, list(range(self.__ngpu)))
        # Apply the ``weights_init`` function to randomly initialize all weights
        #  to ``mean=0``, ``stdev=0.02``.
        self.__generator.apply(weights_init)
        # Print the saved_model
        self.__logger.info(self.__generator)

        # Create the Discriminator
        self.__discriminator = Discriminator(self.__model_params).to(self.__device)
        # Handle multi-GPU if desired
        if (self.__device.type == 'cuda') and (self.__ngpu > 1):
            self.__discriminator = nn.DataParallel(self.__discriminator, list(range(self.__ngpu)))

        # Apply the ``weights_init`` function to randomly initialize all weights
        # like this: ``to mean=0, stdev=0.2``.
        self.__discriminator.apply(weights_init)
        # Print the saved_model
        self.__logger.info(self.__discriminator)

        # Initialize the ``BCELoss`` function
        self.__criterion = nn.BCELoss()
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.__fixed_noise = torch.randn(64, self.__model_params['nz'], 1, 1, device=self.__device)

        # Establish convention for real and fake labels during training
        self.__real_label = 1.
        self.__fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.__optimizerD = optim.Adam(self.__discriminator.parameters(), lr=self.__lr, betas=(self.__beta1, 0.999))
        self.__optimizerG = optim.Adam(self.__generator.parameters(), lr=self.__lr, betas=(self.__beta1, 0.999))

    def __train(self, classify_type: ClassifierType):
        self.__init_dataset(classify_type)
        self.__plot_training_imgs()

        # Training Loop
        iters = 0
        self.__logger.debug('Starting Training Loop ...')
        # For each epoch
        for epoch in range(self.__num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.__dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.__discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.__device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.__real_label, dtype=torch.float, device=self.__device)
                # Forward pass real batch through D
                output = self.__discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.__criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.__model_params['nz'], 1, 1, device=self.__device)
                # Generate fake image batch with G
                fake = self.__generator(noise)
                label.fill_(self.__fake_label)
                # Classify all fake batch with D
                output = self.__discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.__criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.__optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.__generator.zero_grad()
                label.fill_(self.__real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.__discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.__criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.__optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    self.__logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                                       % (epoch, self.__num_epochs, i, len(self.__dataloader),
                                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.__G_losses.append(errG.item())
                self.__D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.__num_epochs - 1) and (i == len(self.__dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.__generator(self.__fixed_noise).detach().cpu()
                    self.__img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
        self.__logger.info('Training DONE!')

    def train(self, classify_type: ClassifierType):

        self.__train(classify_type)
        pass

    def __show_result(self):
        # Loss versus training iteration
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.__G_losses, label="G")
        plt.plot(self.__D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.__img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())

        # Visualization of G's progression
        # Grab a batch of real images from the dataloader
        real_batch = next(iter(self.__dataloader))
        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.__device)[:64], padding=5, normalize=True).cpu(),
                         (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.__img_list[-1], (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    t = Trainner()
    t.train(ClassifierType.rfc)
