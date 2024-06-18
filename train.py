import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from utils.logger import TgaLogger
from model import Generator, Discriminator, weights_init
from utils.classify_type import ClassifierType
from utils.tool import check_fix_path


class Trainner:
    def __init__(self):
        self.__logger = TgaLogger('trainer').get_logger()
        self.__model_params = {
            'nc': 1,  # Number of channels in the training images. For color images this is 3
            'nz': 128,  # Size of z latent vector (i.e. size of generator input)
            'ngf': 64,  # Size of feature maps in generator
            'ndf': 64  # Size of feature maps in discriminator
        }
        self.__workers = 8  # Number of workers for dataloader
        self.__batch_size = 64  # Batch size during training
        self.__image_size = 32  # Spatial size of training images. Images will be resized to this using a transformer.
        # Number of training epochs
        self.__num_epochs = 20  # 20
        self.__g_lr = 0.0002  # Learning rate for optimizers
        self.__d_lr = 0.0002  # 0.00005
        self.__beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers
        self.__ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
        # Decide which device we want to run on
        self.__device = torch.device("cuda:0" if (torch.cuda.is_available() and self.__ngpu > 0) else "cpu")
        self.__data_path = os.path.realpath('./resources/img')
        self.__model_path = os.path.realpath('./saved')

        self.__init_random()

        self.__n_sample = 1e5

        # Lists to keep track of progress
        self.__img_list = None
        self.__G_losses = None
        self.__D_losses = None
        self.__min_G_loss_param = None

        # Establish convention for real and fake labels during training
        self.__real_label = 1.
        self.__fake_label = 0.

        self.__num_epochs_pretrain_G = 1
        self.__num_epochs_pretrain_D = 3
        self.__g_step = 1
        self.__d_step = 1

        self.__alpha = 1  # 判别器损失中真实样本的权重
        self.__beta = 1  # 判别器损失中假样本的权重
        self.__gamma = 1  # 生成器损失的权重

        self.__log_params()

    def __log_params(self):
        self.__logger.debug('#' * 10 + ' Params ' + '#' * 10)
        self.__logger.info(f'workers: {self.__workers}'),
        self.__logger.info(f'batch size: {self.__batch_size}')
        self.__logger.info(f'image size: {self.__image_size}')
        self.__logger.info(f'epochs: {self.__num_epochs}')
        self.__logger.info(f'g learning rate: {self.__g_lr}')
        self.__logger.info(f'd learning rate: {self.__d_lr}')
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

    def __init_dataset(self, data_path):
        self.__logger.debug('Init dataset ...')
        self.__data_root = data_path  # Root directory for dataset
        self.__logger.info(f'Data path: {self.__data_root}')
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        self.__logger.info('Loading data ...')
        # self.__logger.info('Computing mean&std ...')
        # self.__dataset = dset.ImageFolder(root=self.__data_root,
        #                                   transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
        # self.__dataloader = torch.utils.data.DataLoader(self.__dataset, batch_size=self.__batch_size, shuffle=True,
        #                                                 num_workers=self.__workers)
        # self.__data_mean = 0.0
        # self.__data_std = 0.0
        # nb_samples = 0
        #
        # for data in self.__dataloader:
        #     images, _ = data
        #     batch_samples = images.size(0)
        #     images = images.view(batch_samples, -1)
        #     self.__data_mean += images.mean(1).sum(0)
        #     self.__data_std += images.std(1).sum(0)
        #     nb_samples += batch_samples
        #
        # self.__data_mean /= nb_samples
        # self.__data_std /= nb_samples
        #
        # self.__logger.info(f'data_mean: {self.__data_mean}')
        # self.__logger.info(f'data_std: {self.__data_std}')
        # self.__logger.info('Computing mean&std DONE!')

        self.__data_mean = 0
        self.__data_std = 1

        self.__logger.info('Reloading data ...')
        self.__dataset = dset.ImageFolder(root=self.__data_root, transform=transforms.Compose([
            transforms.Grayscale(),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # 归一化 依据实际数据集的均值和方差
            # transforms.Normalize(self.__data_mean, self.__data_std),
        ]))
        # .__subset = Subset(self.__dataset, torch.arange(self.__n_sample))
        self.__logger.info('Loading data DONE!')
        self.__logger.info(f'Data :{self.__dataset.class_to_idx}')
        # Create the dataloader
        self.__dataloader = DataLoader(self.__dataset, batch_size=self.__batch_size, shuffle=True,
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

        # Setup Adam optimizers for both G and D
        self.__optimizerD = optim.Adam(self.__discriminator.parameters(), lr=self.__d_lr, betas=(self.__beta1, 0.999))
        self.__optimizerG = optim.Adam(self.__generator.parameters(), lr=self.__g_lr, betas=(self.__beta1, 0.999))

    def __train(self):
        self.__init_model_loss_opt()

        # Pretrain the generator
        self.__logger.debug('Pretrain the generator ...')
        for epoch in range(self.__num_epochs_pretrain_G):
            for i, data in enumerate(self.__dataloader, 0):
                self.__generator.zero_grad()
                b_size = data[0].size(0)
                noise = torch.randn(b_size, self.__model_params['nz'], 1, 1, device=self.__device)
                fake = self.__generator(noise)
                label = torch.full((b_size,), self.__real_label, dtype=torch.float, device=self.__device)
                output = self.__discriminator(fake).view(-1)
                errG = self.__criterion(output, label)
                errG.backward()
                self.__optimizerG.step()
                if i % 50 == 0:
                    self.__logger.debug(f'[{epoch + 1}/{self.__num_epochs_pretrain_G}][{i}/{len(self.__dataloader)}]'
                                        f'Loss_G: {errG.item()}')

        # Pretrain the discriminator
        self.__logger.debug('Pretrain the discriminator ...')
        for epoch in range(self.__num_epochs_pretrain_D):
            for i, data in enumerate(self.__dataloader, 0):
                # train with real batch
                self.__discriminator.zero_grad()
                real_data = data[0].to(self.__device)
                b_size = real_data.size(0)
                label = torch.full((b_size,), self.__real_label, dtype=torch.float, device=self.__device)
                output = self.__discriminator(real_data).view(-1)
                errD_real = self.__criterion(output, label)
                errD_real.backward()

                # train with fake batch
                noise = torch.randn(b_size, self.__model_params['nz'], 1, 1, device=self.__device)
                fake = self.__generator(noise)
                label.fill_(self.__fake_label)
                output = self.__discriminator(fake.detach()).view(-1)
                errD_fake = self.__criterion(output, label)
                errD_fake.backward()
                self.__optimizerD.step()

                errD = errD_real + errD_fake
                if i % 50 == 0:
                    self.__logger.debug(f'[{epoch + 1}/{self.__num_epochs_pretrain_D}][{i}/{len(self.__dataloader)}]'
                                        f'Loss_D: {errD.item()}')

        # Training Loop
        iters = 0
        last_loss_g = 100
        self.__logger.debug('Starting Training Loop ...')
        # For each epoch
        for epoch in range(self.__num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.__dataloader, 0):
                for _ in range(self.__d_step):
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
                    errD_real = self.__alpha * errD_real
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
                    errD_fake = self.__beta * errD_fake
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    if errD < 2.5:
                        self.__optimizerD.step()

                for _ in range(self.__g_step):
                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    self.__generator.zero_grad()
                    label.fill_(self.__real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = self.__discriminator(fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = self.__criterion(output, label)
                    errG = self.__gamma * errG
                    # Calculate gradients for G
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    # Update G
                    self.__optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    self.__logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                                       % (epoch + 1, self.__num_epochs, i, len(self.__dataloader),
                                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # save best model
                if errG.item() < last_loss_g:
                    self.__min_G_loss_param = {
                        'model_params': self.__model_params,
                        'generator': self.__generator.state_dict(),
                        'discriminator': self.__discriminator.state_dict(),
                        'optimizerG': self.__optimizerG.state_dict(),
                        'optimizerD': self.__optimizerD.state_dict(),
                        'dataset_mean': self.__data_mean,
                        'dataset_std': self.__data_std,
                    }
                    last_loss_g = errG.item()
                # Save Losses for plotting later
                self.__G_losses.append(errG.item())
                self.__D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.__num_epochs) and (i == len(self.__dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.__generator(self.__fixed_noise).detach().cpu()
                    self.__img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
        self.__logger.info('Training DONE!')

    def train(self, classify_type: ClassifierType):
        self.__data_path = os.path.join(self.__data_path, classify_type.value)
        for _t in os.listdir(self.__data_path):
            if len(os.listdir(os.path.join(self.__data_path, _t, 'src'))) < 1000:
                self.__logger.warning(f'Too few data: {os.path.join(self.__data_path, _t)}')
                continue
            self.__img_list = []
            self.__G_losses = []
            self.__D_losses = []
            self.__logger.debug(f'Train {classify_type.value}_{_t}')
            data_path = os.path.join(self.__data_path, _t)
            self.__init_dataset(data_path)
            self.__plot_training_imgs()
            self.__train()
            self.__show_result()
            model_path = os.path.join(self.__model_path, classify_type.value, _t)
            check_fix_path(model_path)
            self.__save_model(model_path)

    def __save_model(self, model_path):
        best_model_name = os.path.join(model_path, 'model_bestG.pth')
        self.__logger.debug(f'Saving {best_model_name} ...')
        torch.save(self.__min_G_loss_param, best_model_name)
        self.__logger.debug(f'Save {best_model_name} DONE!')

        final_model_name = os.path.join(model_path, 'model_final.pth')
        self.__logger.debug(f'Saving {final_model_name} ...')
        torch.save({
            'model_params': self.__model_params,
            'generator': self.__generator.state_dict(),
            'discriminator': self.__discriminator.state_dict(),
            'optimizerG': self.__optimizerG.state_dict(),
            'optimizerD': self.__optimizerD.state_dict(),
            'dataset_mean': self.__data_mean,
            'dataset_std': self.__data_std,
        }, final_model_name)
        self.__logger.debug(f'Save {final_model_name} DONE!')

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
