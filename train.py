import json
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
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from utils.logger import TgaLogger
from model import DCGAN_G
from model import DCGAN_D
from model import DCGAN_G_nobn
from model import DCGAN_D_nobn
from model import weights_init
from utils.classify_type import ClassifierType
from utils.tool import check_fix_path,check_del_path
from torch.autograd import Variable


class Trainner:
    def __init__(self):
        self.__logger = TgaLogger('trainer').get_logger()
        self.__model_params = {
            'nc': 1,  # Number of channels in the training images. For color images this is 3
            'nz': 512,  # Size of z latent vector (i.e. size of generator input)
            'ngf': 64,  # 64 # Size of feature maps in generator
            'ndf': 64,  # 64 #   Size of feature maps in discriminator
            'n_extra_layers': 0  # 额外层数
        }
        self.__workers = 8  # Number of workers for dataloader
        self.__batch_size = 64  # Batch size during training
        self.__image_size = 32  # Spatial size of training images. Images will be resized to this using a transformer.
        # Number of training epochs
        self.__num_epochs = 50  # 20
        self.__g_lr = 0.0001  # Learning rate for optimizers
        self.__d_lr = 0.0001  # 0.00005
        self.__beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers
        self.__ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
        self.__noBN = False
        # Decide which device we want to run on
        self.__device = torch.device("cuda:0" if (torch.cuda.is_available() and self.__ngpu > 0) else "cpu")
        self.__cuda = True
        cudnn.benchmark = True

        self.__data_path = os.path.realpath('./resources/img')
        self.__model_path = os.path.realpath('./saved')

        self.__init_random()

        self.__n_sample = 1e5

        # Lists to keep track of progress
        self.__G_losses = None
        self.__D_losses = None
        self.__min_G_loss_param = None

        # Establish convention for real and fake labels during training
        self.__real_label = 1.
        self.__fake_label = 0.

        self.__num_epochs_pretrain_G = 20  # 2
        self.__num_epochs_pretrain_D = 0  # 1
        self.__g_step = 1
        self.__d_step = 1
        self.__dpg = 5  # number of D iters per each G iter

        self.__clamp_lower = -0.01
        self.__clamp_upper = 0.01

        self.__g_threshold = 0  # 0.7
        self.__d_threshold = 0  # 0.8

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

        self.__data_mean = 0.5
        self.__data_std = 0.5

        self.__logger.info('Reloading data ...')
        self.__dataset = dset.ImageFolder(root=self.__data_root, transform=transforms.Compose([
            transforms.Grayscale(),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize(self.__data_mean, self.__data_std),
        ]))
        # .__subset = Subset(self.__dataset, torch.arange(self.__n_sample))
        self.__logger.info('Loading data DONE!')
        self.__logger.info(f'Data :{self.__dataset.class_to_idx}')
        # Create the dataloader
        self.__dataloader = DataLoader(self.__dataset, batch_size=self.__batch_size, shuffle=True,
                                       num_workers=self.__workers)

    def __init_model_loss_opt(self, save_path):
        self.__logger.debug('Init model, loss and opt ...')

        # write out generator config to generate images together wth training checkpoints (.pth)
        generator_config = {'imageSize': self.__image_size, 'nz': self.__model_params['nz'],
                            'nc': self.__model_params['nc'], 'ngf': self.__model_params['ngf'], 'ngpu': self.__ngpu,
                            'n_extra_layers': self.__model_params['n_extra_layers'], 'noBN': self.__noBN}
        with open(os.path.join(save_path, "generator_config.json"), 'w') as gcfg:
            gcfg.write(json.dumps(generator_config) + '\n')
        self.__logger.info('Write generator config DONE!')

        # Create the generator
        if self.__noBN:
            self.__generator = DCGAN_G_nobn(self.__image_size, self.__model_params['nz'], self.__model_params['nc'],
                                            self.__model_params['ngf'], self.__ngpu,
                                            self.__model_params['n_extra_layers'])
        else:
            self.__generator = DCGAN_G(self.__image_size, self.__model_params['nz'], self.__model_params['nc'],
                                       self.__model_params['ngf'], self.__ngpu, self.__model_params['n_extra_layers'])
        self.__generator = self.__generator

        # Handle multi-GPU if desired
        if (self.__device.type == 'cuda') and (self.__ngpu > 1):
            self.__generator = nn.DataParallel(self.__generator, list(range(self.__ngpu)))
        # Apply the ``weights_init`` function to randomly initialize all weights
        self.__generator.apply(weights_init)
        # Print the saved_model
        self.__logger.info(self.__generator)

        # Create the Discriminator
        self.__discriminator = DCGAN_D(self.__image_size, self.__model_params['nz'], self.__model_params['nc'],
                                       self.__model_params['ndf'], self.__ngpu, self.__model_params['n_extra_layers'])
        self.__discriminator = self.__discriminator

        # Handle multi-GPU if desired
        if (self.__device.type == 'cuda') and (self.__ngpu > 1):
            self.__discriminator = nn.DataParallel(self.__discriminator, list(range(self.__ngpu)))

        # Apply the ``weights_init`` function to randomly initialize all weights
        self.__discriminator.apply(weights_init)
        # Print the saved_model
        self.__logger.info(self.__discriminator)

        # Initialize the ``BCELoss`` function
        # self.__criterion = nn.BCELoss()
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.__fixed_noise = torch.randn(64, self.__model_params['nz'], 1, 1, device=self.__device)

        # Setup Adam optimizers for both G and D
        # self.__optimizerD = optim.Adam(self.__discriminator.parameters(), lr=self.__d_lr, betas=(self.__beta1, 0.999))
        # self.__optimizerG = optim.Adam(self.__generator.parameters(), lr=self.__g_lr, betas=(self.__beta1, 0.999))

        # Setup RMSprop optimizers for both G and D
        self.__optimizerD = optim.RMSprop(self.__discriminator.parameters(), lr=self.__d_lr)
        self.__optimizerG = optim.RMSprop(self.__generator.parameters(), lr=self.__g_lr)

    def __train(self, save_path):
        self.__init_model_loss_opt(save_path)

        input = torch.FloatTensor(self.__batch_size, self.__model_params['nc'], self.__image_size, self.__image_size)
        noise = torch.FloatTensor(self.__batch_size, self.__model_params['nz'], 1, 1)
        fixed_noise = torch.FloatTensor(self.__batch_size, self.__model_params['nz'], 1, 1).normal_(0, 1)
        one = torch.FloatTensor([1])
        mone = one * -1

        if self.__cuda:
            self.__discriminator.cuda()
            self.__generator.cuda()
            input = input.cuda()
            one, mone = one.cuda(), mone.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        for epoch in range(self.__num_epochs):
            data_iter = iter(self.__dataloader)
            i = 0
            while i < len(self.__dataloader):
                ############################
                # (1) Update D network
                ###########################
                for p in self.__discriminator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                # train the discriminator Diters times
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    Diters = 100
                else:
                    Diters = self.__dpg
                j = 0
                while j < Diters and i < len(self.__dataloader):
                    j += 1

                    # clamp parameters to a cube
                    for p in self.__discriminator.parameters():
                        p.data.clamp_(self.__clamp_lower, self.__clamp_upper)

                    data = next(data_iter)  # .next()
                    i += 1

                    # train with real
                    real_cpu, _ = data
                    self.__discriminator.zero_grad()
                    batch_size = real_cpu.size(0)

                    if self.__cuda:
                        real_cpu = real_cpu.cuda()
                    input.resize_as_(real_cpu).copy_(real_cpu)
                    inputv = Variable(input)

                    errD_real = self.__discriminator(inputv)
                    errD_real.backward(one)

                    # train with fake
                    noise.resize_(self.__batch_size, self.__model_params['nz'], 1, 1).normal_(0, 1)
                    with torch.no_grad():
                        noisev = Variable(noise)
                    # noisev = Variable(noise, volatile = True) # totally freeze netG
                    fake = Variable(self.__generator(noisev).data)
                    inputv = fake
                    errD_fake = self.__discriminator(inputv)
                    errD_fake.backward(mone)
                    errD = errD_real - errD_fake
                    self.__optimizerD.step()

                ############################
                # (2) Update G network
                ###########################
                for p in self.__discriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                self.__generator.zero_grad()
                # in case our last batch was the tail batch of the dataloader,
                # make sure we feed a full batch of noise
                noise.resize_(self.__batch_size, self.__model_params['nz'], 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = self.__generator(noisev)
                errG = self.__discriminator(fake)
                errG.backward(one)
                self.__optimizerG.step()
                gen_iterations += 1

                self.__logger.info('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                                   % (epoch + 1, self.__num_epochs, i, len(self.__dataloader), gen_iterations,
                                      errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
                if gen_iterations % 500 == 0:
                    real_cpu = real_cpu.mul(0.5).add(0.5)
                    vutils.save_image(real_cpu, '{0}/real_samples.png'.format(save_path))
                    # fake = self.__generator(Variable(fixed_noise, volatile=True))
                    with torch.no_grad():
                        fake = self.__generator(Variable(fixed_noise))
                    fake.data = fake.data.mul(0.5).add(0.5)
                    vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(save_path, gen_iterations))

                self.__G_losses.append(errG.item())
                self.__D_losses.append(errD.item())

            # do checkpointing
            self.__logger.debug('Saving  {0}/netG_epoch_{1}.pth'.format(save_path, epoch))
            torch.save(self.__generator.state_dict(), '{0}/netG_epoch_{1}.pth'.format(save_path, epoch))
            # torch.save(self.__discriminator.state_dict(), '{0}/netD_epoch_{1}.pth'.format(save_path, epoch))
            self.__logger.debug('Save DONE!')

        self.__logger.info('Training DONE!')

    def train(self, classify_type: ClassifierType):
        self.__data_path = os.path.join(self.__data_path, classify_type.value)
        for _t in os.listdir(self.__data_path):
            # if len(os.listdir(os.path.join(self.__data_path, _t, 'src'))) < 1000:
            #     self.__logger.warning(f'Too few data: {os.path.join(self.__data_path, _t)}')
            #     continue
            self.__G_losses = []
            self.__D_losses = []
            self.__logger.debug(f'Train {classify_type.value}_{_t}')
            data_path = os.path.join(self.__data_path, _t)
            save_path = os.path.join(self.__model_path, classify_type.value, _t)
            check_del_path(save_path)
            check_fix_path(save_path)
            self.__init_dataset(data_path)
            self.__train(save_path)
            self.__show_loss()

    def __show_loss(self):
        # Loss versus training iteration
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.__G_losses, label="G")
        plt.plot(self.__D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    t = Trainner()
    t.train(ClassifierType.eip)
