import json
import os
import random

import torch
from utils.logger import TgaLogger
from utils.classify_type import ClassifierType
from utils.tool import check_fix_path
import numpy as np
from model import DCGAN_G
from model import DCGAN_G_nobn
from model import weights_init


class IPv6Generator(object):
    def __init__(self):
        self.__load_path = os.path.realpath('./saved')
        self.__out_path = os.path.realpath('./output')
        self.__logger = TgaLogger('generator').get_logger()

        self.__device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.__cuda = True

        self.__n_image_gen = 10000  # int(math.ceil(64000 / 32))

    # def __gen(self, model_path) -> list:
    #     # Init
    #     self.__logger.debug(f'Init IPv6Generator in {model_path} ...')
    #     self.__status_dict = torch.load(model_path)
    #     self.__v6generator = Generator(self.__status_dict['model_params']).to(self.__device)
    #     self.__v6generator.load_state_dict(self.__status_dict['generator'])
    #     self.__logger.info(self.__v6generator)
    #     self.__logger.debug('Init IPv6Generator DONE!')
    #
    #     self.__logger.debug('Gen address with IPv6Generator ...')
    #     # Get latent vector Z from unit normal distribution.
    #     noise = torch.randn(self.__n_image_gen, self.__status_dict['model_params']['nz'], 1, 1, device=self.__device)
    #     # Turn off gradient calculation to speed up the process.
    #     with torch.no_grad():
    #         # Get generated image from the noise vector using
    #         # the trained generator.
    #         generated_imgs = self.__v6generator(noise).detach().cpu()
    #         self.__logger.info('IMG generated')
    #
    #     # Covert generated image to addresses.
    #     self.__logger.debug('Covering generated image to addresses ...')
    #     # 反归一化
    #     generated_imgs = ((generated_imgs * self.__status_dict['dataset_std']) + self.__status_dict['dataset_mean']) * 255
    #     #generated_imgs= generated_imgs * 255
    #
    #     address_sequences = []
    #     l = len(generated_imgs)
    #     n = 0
    #     for img in generated_imgs:
    #         # 对每张图片的每一行进行处理
    #         img=img.squeeze(0)
    #         for row in img:
    #             # chu17并四舍五入，然后转为整型
    #             sqrt_row_int = torch.round(row/17).to(torch.int).tolist()
    #             # 将每个元素转换为一位16进制字符串
    #             # hex_list = [ f'{x:x}' for x in sqrt_row_int]
    #             hex_list=[]
    #             for x in sqrt_row_int:
    #                 if x<0:
    #                     hex_list.append('0')
    #                 elif x>15:
    #                     hex_list.append('f')
    #                 else:
    #                     hex_list.append(f'{x:x}')
    #             # 将每四个元素分为一组
    #             grouped_hex = [''.join(hex_list[i:i + 4]) for i in range(0, len(hex_list), 4)]
    #             # 将所有组用冒号连接成最终字符串
    #             address = ':'.join(grouped_hex)
    #             n += 1
    #             self.__logger.info(f'Generated address {n}/{l * 32}: {address}')
    #             # 将处理后的行添加到总的地址序列列表中
    #             address_sequences.append(address)
    #
    #     return address_sequences
    @staticmethod
    def __imag2addr_1(img) -> str:
        img = img*255
        img = img.squeeze(0)
        # chu17并四舍五入，然后转为整型
        row = torch.round(img.mean(dim=0) / 17).to(torch.int).tolist()
        hex_list = []
        for x in row:
            if x < 0:
                hex_list.append('0')
            elif x > 15:
                hex_list.append('f')
            else:
                hex_list.append(f'{x:x}')
        # 将每四个元素分为一组
        grouped_hex = [''.join(hex_list[i:i + 4]) for i in range(0, len(hex_list), 4)]
        # 将所有组用冒号连接成最终字符串
        address = ':'.join(grouped_hex)
        return address

    @staticmethod
    def __image2addr_2(img):
        img = img * 255
        img = img.squeeze(0)
        for i in range(1, 32, 2):
            img[i] = torch.flip(img[i], dims=[0])
        row = torch.round(img.mean(dim=0) / 17).to(torch.int).tolist()
        hex_list = []
        for x in row:
            if x < 0:
                hex_list.append('0')
            elif x > 15:
                hex_list.append('f')
            else:
                hex_list.append(f'{x:x}')
        grouped_hex = [''.join(hex_list[i:i + 4]) for i in range(0, len(hex_list), 4)]
        address = ':'.join(grouped_hex)
        return address

    @staticmethod
    def __image2addr_3(img):
        img = img.squeeze(0)
        img = img.tolist()
        hex_list = []
        for col in range(32):
            col_r = []
            for i in range(0, 32, 2):
                if i + 1 < 32:
                    sum = img[i][col] + img[i + 1][col]
                    col_r.append(sum)
            if max(col_r) < 0.6:  # 采样阈值
                hex_list.append(f'{random.randint(0, 15):x}')
            else:
                hex_list.append(f'{np.argmax(np.array(col_r)):x}')
        grouped_hex = [''.join(hex_list[i:i + 4]) for i in range(0, len(hex_list), 4)]
        address = ':'.join(grouped_hex)
        return address

    def __gen(self, model_path, epoch) -> list:
        # Init
        self.__logger.debug(f'Init IPv6Generator in {model_path} ...')
        conf = os.path.join(model_path, 'generator_config.json')
        with open(conf, 'r') as gencfg:
            generator_config = json.loads(gencfg.read())

        imageSize = generator_config["imageSize"]
        nz = generator_config["nz"]
        nc = generator_config["nc"]
        ngf = generator_config["ngf"]
        noBN = generator_config["noBN"]
        ngpu = generator_config["ngpu"]
        n_extra_layers = generator_config["n_extra_layers"]

        if noBN:
            self.__v6generator = DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
        else:
            self.__v6generator = DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

        # load weights
        mpth = os.path.join(model_path, f'netG_epoch_{epoch}.pth')
        self.__v6generator.load_state_dict(torch.load(mpth))

        self.__logger.info(self.__v6generator)
        self.__logger.debug('Init IPv6Generator DONE!')

        self.__logger.debug('Gen address with IPv6Generator ...')
        # initialize noise
        fixed_noise = torch.FloatTensor(self.__n_image_gen, nz, 1, 1).normal_(0, 1)

        if self.__cuda:
            self.__v6generator.cuda()
            fixed_noise = fixed_noise.cuda()

        fake = self.__v6generator(fixed_noise)

        self.__logger.info('IMG generated')
        # save img_sample
        # for i in range(10):
        #     vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(opt.output_dir, "generated_%02d.png"%i))
        # Covert generated image to addresses.
        self.__logger.debug('Covering generated image to addresses ...')
        # 反归一化
        generated_imgs = fake.data.mul(0.5).add(0.5)

        address_sequences = []
        l = len(generated_imgs)
        n = 0
        for img in generated_imgs:
            address = self.__image2addr_3(img)
            n += 1
            self.__logger.info(f'Generated address {n}/{l}: {address}')
            # 将处理后的行添加到总的地址序列列表中
            address_sequences.append(address)
        self.__logger.debug(f'地址数: {len(address_sequences)}')
        address_sequences = list(set(address_sequences))
        self.__logger.debug(f'去重地址数: {len(address_sequences)}')

        while len(address_sequences) < self.__n_image_gen:
            self.__logger.debug(f'未达到指定数量{self.__n_image_gen}，继续生成')
            # initialize noise
            fixed_noise = torch.FloatTensor(500, nz, 1, 1).normal_(0, 1)

            if self.__cuda:
                fixed_noise = fixed_noise.cuda()

            fake = self.__v6generator(fixed_noise)
            generated_imgs = fake.data.mul(0.5).add(0.5)
            l = len(generated_imgs)
            n = 0
            for img in generated_imgs:
                address = self.__image2addr_3(img)
                n += 1
                self.__logger.info(f'Continuing generate address: {address}')
                address_sequences.append(address)
            address_sequences = list(set(address_sequences))

        if len(address_sequences) > self.__n_image_gen:
            self.__logger.debug(f'共生成地址 {len(address_sequences)} 个，截断')
            address_sequences = address_sequences[:self.__n_image_gen]

        self.__logger.debug('Gen addresses DONE!')

        return address_sequences

    def gen(self, c_type: ClassifierType, epoch: int = 99, *, num_gen=None):
        if num_gen is not None:
            self.__n_image_gen = num_gen  # // int(math.ceil(num_gen / 32))
        c_path = os.path.join(self.__load_path, c_type.value)
        for _t in os.listdir(c_path):
            m_path = os.path.join(c_path, _t)
            addresses = self.__gen(m_path, epoch)
            o_path = os.path.join(self.__out_path, c_type.value, _t)
            check_fix_path(o_path)
            with open(os.path.join(o_path, f'epoch_{epoch}.txt'), 'w', encoding='utf-8') as o:
                o.writelines('\n'.join(addresses))


if __name__ == '__main__':
    gen = IPv6Generator()
    gen.gen(ClassifierType.eip, epoch=49, num_gen=10000)
