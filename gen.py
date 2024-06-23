import math
import os
import torch
from utils.logger import TgaLogger
from model import Generator
from utils.classify_type import ClassifierType
from utils.tool import check_fix_path
import numpy as np


class IPv6Generator(object):
    def __init__(self):
        self.__load_path = os.path.realpath('./saved')
        self.__out_path = os.path.realpath('./output')
        self.__logger = TgaLogger('generator').get_logger()

        self.__device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.__n_image_gen = 32000  # int(math.ceil(64000 / 32))

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

    def __gen(self, model_path) -> list:
        # Init
        self.__logger.debug(f'Init IPv6Generator in {model_path} ...')
        self.__status_dict = torch.load(model_path)
        self.__v6generator = Generator(self.__status_dict['model_params']).to(self.__device)
        self.__v6generator.load_state_dict(self.__status_dict['generator'])
        self.__logger.info(self.__v6generator)
        self.__logger.debug('Init IPv6Generator DONE!')

        self.__logger.debug('Gen address with IPv6Generator ...')
        # Get latent vector Z from unit normal distribution.
        noise = torch.randn(self.__n_image_gen, self.__status_dict['model_params']['nz'], 1, 1, device=self.__device)
        # Turn off gradient calculation to speed up the process.
        with torch.no_grad():
            # Get generated image from the noise vector using
            # the trained generator.
            generated_imgs = self.__v6generator(noise).detach().cpu()
            self.__logger.info('IMG generated')

        # Covert generated image to addresses.
        self.__logger.debug('Covering generated image to addresses ...')
        # 反归一化
        generated_imgs = ((generated_imgs * self.__status_dict['dataset_std']) + self.__status_dict[
            'dataset_mean']) * 255
        # generated_imgs= generated_imgs * 255

        address_sequences = []
        l = len(generated_imgs)
        n = 0
        for img in generated_imgs:
            # 对每张图片的每一行进行处理
            img = img.squeeze(0)
            img = img.tolist()

            # row = torch.round(img.mean(dim=0) / 17).to(torch.int).tolist()

            # for i in range(1, 32, 2):
            #     img[i] = torch.flip(img[i], dims=[0])
            # row = torch.round(img.mean(dim=0) / 17).to(torch.int).tolist()

            hex_list = []
            for col in range(32):
                col_r = []
                for i in range(0, 32, 2):
                    if i + 1 < 32:
                        sum = img[i][col] + img[i + 1][col]
                        col_r.append(sum)

                hex_list.append(f'{np.argmax(np.array(col_r)):x}')


            # for x in row:
            #     if x < 0:
            #         hex_list.append('0')
            #     elif x > 15:
            #         hex_list.append('f')
            #     else:
            #         hex_list.append(f'{x:x}')
            # 将每四个元素分为一组
            grouped_hex = [''.join(hex_list[i:i + 4]) for i in range(0, len(hex_list), 4)]
            # 将所有组用冒号连接成最终字符串
            address = ':'.join(grouped_hex)
            n += 1
            self.__logger.info(f'Generated address {n}/{l}: {address}')
            # 将处理后的行添加到总的地址序列列表中
            address_sequences.append(address)

        return address_sequences

    def gen(self, c_type: ClassifierType, model_type: str = 'final', *, num_gen=None):
        if num_gen is not None:
            self.__n_image_gen = num_gen  # // int(math.ceil(num_gen / 32))
        c_path = os.path.join(self.__load_path, c_type.value)
        for _t in os.listdir(c_path):
            m_name = None
            if model_type == 'final':
                m_name = os.path.join(c_path, _t, 'model_final.pth')
            elif model_type == 'bestG':
                m_name = os.path.join(c_path, _t, 'model_bestG.pth')
            addresses = self.__gen(m_name)
            o_path = os.path.join(self.__out_path, c_type.value)
            check_fix_path(o_path)
            with open(os.path.join(o_path, _t + '.txt'), 'w', encoding='utf-8') as o:
                o.writelines('\n'.join(addresses))


if __name__ == '__main__':
    gen = IPv6Generator()
    gen.gen(ClassifierType.rfc, num_gen=10000)  # ,'bestG')
