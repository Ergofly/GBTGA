import numpy as np
from utils.classify_type import ClassifierType
import os
import warnings
from PIL import Image
from utils.logger import TgaLogger
from utils.tool import check_fix_path, check_del_path, IPv62VecCluster
import shutil
import random


class Classifier:
    """
    Classifier is a class that is used to classify the seed
    """

    def __init__(self, data_path, seeds_path, tool_path, *, log_name='classifier'):
        self.logger = TgaLogger(log_name).get_logger()
        if data_path is None:
            self.logger.error('data_path is None')
            raise ValueError('data_path is None')
        if seeds_path is None:
            self.logger.error('seeds_path is None')
            raise ValueError('seeds_path is None')
        self.data_path = os.path.realpath(data_path)
        self.seeds_path = os.path.realpath(seeds_path)
        self.tool_path = os.path.realpath(tool_path)
        self.out_path = os.path.join(self.data_path, 'classify_data')
        check_fix_path(self.out_path)

    def classify(self):
        pass


class RandomizedSampleClassifier(Classifier):
    def __init__(self, data_path, seeds_path, tool_path, *, log_name='randomized_sample_classifier', n_sample=0):
        super().__init__(data_path, seeds_path, tool_path, log_name=log_name)
        self.__logger = self.logger
        self.__out_path = os.path.join(self.out_path, 'randomized_sample')
        check_del_path(self.__out_path)
        check_fix_path(self.__out_path)
        if n_sample > 0:
            self.__n_sample = n_sample

    def classify(self):
        if self.__n_sample == 0:
            self.__logger.info('采样数 0 ，直接复制')
            shutil.copy2(self.seeds_path, os.path.join(self.__out_path, f'randomized_sample.txt'))
            return None
        with open(self.seeds_path, 'r') as file:
            lines = file.readlines()
            if self.__n_sample > len(lines):
                self.__logger.error("采样的行数超过了总地址数")
                raise ValueError("采样的行数超过了总地址数")
            self.__logger.info(f'Sample {self.__n_sample} rows')
            sampled_lines = random.sample(lines, self.__n_sample)
            self.__logger.info('Sample DONE!')
        with open(os.path.join(self.__out_path, 'randomized_sample.txt'), 'w') as of:
            of.writelines(sampled_lines)
            self.__logger.info('Write to file DONE!')
        return self.__out_path


class RFCClassifier(Classifier):

    def __init__(self, data_path, seeds_path, tool_path, *, log_name='rfc_classifier'):
        super().__init__(data_path, seeds_path, tool_path, log_name=log_name)
        self.__logger = self.logger
        self.__out_path = os.path.join(self.out_path, 'rfc')
        check_del_path(self.__out_path)
        check_fix_path(self.__out_path)
        self.__temp = os.path.join(self.data_path, 'temp')
        check_fix_path(self.__temp)
        self.__rfc_profile = os.path.join(self.__temp, 'rfc_profile.txt')
        self.__classify_dict = {}
        self.__cmd = f'cat "{self.seeds_path}" | "{self.tool_path}/addr6" -i -d > "{self.__rfc_profile}"'

    def classify(self):
        self.__logger.info(f'running cmd: {self.__cmd}...')
        os.system(self.__cmd)
        self.__logger.info('running cmd: success')
        t = open(self.__rfc_profile, 'r')
        s = open(self.seeds_path, 'r')
        self.__logger.info(f'classifying...')
        for line, data in zip(t, s):
            label = line.split('=')[3]
            if label not in self.__classify_dict.keys():
                self.__classify_dict[label] = []
            self.__classify_dict[label].append(data)
        s.close()
        t.close()
        for _t in self.__classify_dict.keys():
            with open(os.path.join(self.__out_path, _t + '.txt'), 'w') as f:
                f.writelines(self.__classify_dict[_t])
        self.logger.info(f'classify dict: {self.__classify_dict.keys()}')
        self.__logger.info(f'classify success. DATA saved at {self.__out_path}')
        return self.__out_path


class IPv62VecClassifier(Classifier):
    def __init__(self, data_path, seeds_path, tool_path, *, log_name='ipv62vec_classifier'):
        super().__init__(data_path, seeds_path, tool_path, log_name=log_name)
        self.__logger = self.logger
        self.__out_path = os.path.join(self.out_path, 'ipv62vec')
        check_del_path(self.__out_path)
        check_fix_path(self.__out_path)
        self.__temp = os.path.join(self.data_path, 'temp')
        check_fix_path(self.__temp)
        self.__ipv6_vec_profile = os.path.join(self.__temp, 'ipv62vec_profile.txt')

    def classify(self):
        cluter = IPv62VecCluster(self.seeds_path, self.__ipv6_vec_profile, self.__out_path)
        cluter.create_category()
        return self.__out_path


class EntropyIPClassifier(Classifier):
    def __init__(self, data_path, seeds_path, tool_path, *, log_name='entropy_ip_classifier'):
        super().__init__(data_path, seeds_path, tool_path, log_name=log_name)
        self.__logger = self.logger
        self.__out_path = os.path.join(self.out_path, 'entropy-ip')
        check_del_path(self.__out_path)
        check_fix_path(self.__out_path)
        self.__temp = os.path.join(self.data_path, 'temp')
        check_fix_path(self.__temp)
        self.__formated_seed = os.path.join(self.__temp, 'formated_seeds.txt')
        self.__eip_profile = os.path.join(self.__temp, 'entropy-ip_profile.txt')
        self.__eip_cluster = os.path.join(self.__temp, 'ec_cluster.txt')
        self.__k = 4
        self.__profile_cmd = f'cat {self.__formated_seed} | ' + \
                             f'{self.tool_path}/entropy-clustering/profiles > {self.__eip_profile}'
        self.__cluster_cmd = f'cat {self.__eip_profile} | ' + \
                             f'{self.tool_path}/entropy-clustering/clusters -kmeans -k ' \
                             + str(self.__k) + f' > {self.__eip_cluster}'


    def classify(self):
        lines = []
        with open(self.seeds_path, 'r') as s:
            for line in s:
                line = line.strip()
                if line == '':
                    continue
                line = line.replace(':', '')
                lines.append(line + '\n')
        with open(self.__formated_seed, 'w') as f:
            f.writelines(lines)
        self.__logger.info('running cmd ...')
        classify_dict = {}
        classify_prefix_dict = {}
        os.system(self.__profile_cmd)
        os.system(self.__cluster_cmd)
        self.__logger.info('running cmd: success')
        for i in range(self.__k):
            classify_dict[str(i)] = []
            classify_prefix_dict[str(i)] = []
        type_pointer = 0
        class_prefix = []
        f = open(self.__eip_cluster, 'r')
        for line in f:
            if line[0] == '=':
                class_prefix = []
            elif line[0] == '\n':
                classify_prefix_dict[str(type_pointer)].extend(class_prefix)
                type_pointer += 1
            elif line[:7] == 'SUMMARY':
                break
            else:
                class_prefix.append(line[:8])
        f.close()

        for type in classify_prefix_dict.keys():
            for prefix in classify_prefix_dict[type]:
                f = open(self.seeds_path, 'r')
                for line in f:
                    if prefix == line.replace(':','')[:8]:
                        classify_dict[type].append(line)
                f.close()

        self.__logger.info('writing files ...')
        for type in classify_dict.keys():
            f = open(os.path.join(self.__out_path, 'cluster_' + type + '.txt'), 'w')
            f.writelines(classify_dict[type])
            f.close()
        self.__logger.info('writing files D0NE!')

        return self.__out_path


class DataProcessor:
    """
    DataProcessor is a class that is used to process the data
    """
    __logger = None
    __root_path = os.path.realpath(os.path.curdir)
    __seeds = list()

    def __init__(self, root_path=None, *, log_name='data_process'):
        self.__logger = TgaLogger(log_name).get_logger()
        if root_path is not None:
            self.__root_path = os.path.realpath(root_path)
        self.__tool_path = os.path.join(self.__root_path, 'v6tools')
        self.__resource_path = os.path.join(self.__root_path, 'resources')
        self.__data_path = os.path.join(self.__resource_path, 'data')
        self.__img_path = os.path.join(self.__resource_path, 'img')
        self.__alias_path = os.path.join(self.__resource_path, 'alias')
        for p in self.__root_path, self.__resource_path, self.__data_path, self.__img_path, self.__alias_path:
            check_fix_path(p)
        self.__seeds_path = os.path.join(self.__data_path, 'seeds.txt')
        self.__fixed_seeds_path = os.path.join(self.__data_path, 'fixed_seeds.txt')
        # self.__seeds_path = os.path.join(self.__data_path, 'old_seeds.txt')
        # self.__fixed_seeds_path = os.path.join(self.__data_path, 'old_fixed_seeds.txt')
        self.__alias_path = os.path.join(self.__alias_path, 'aliased-prefixes.txt')
        if not os.path.exists(self.__seeds_path) or not os.path.exists(self.__alias_path):
            warnings.warn('No seeds or alias found, please check the data folder')
        if not os.path.exists(self.__fixed_seeds_path):
            # 转为ipv6地址完全形式
            os.system(f'cat "{self.__seeds_path}" | "{self.__tool_path}/addr6" -i -f > "{self.__fixed_seeds_path}"')

    def sample_source(self, sample_n=50000):
        sample_path = os.path.join(self.__data_path, f'seeds_sample_{sample_n}.txt')
        if os.path.exists(sample_path):
            self.__logger.info(f'Sample_{sample_n}  already exists')
        else:
            self.__logger.info(f'Sampling {sample_n} seeds...')
            with open(self.__fixed_seeds_path, 'r') as f:
                seeds = f.readlines()
                seeds = random.sample(seeds, sample_n)
                with open(sample_path, 'w') as f:
                    f.writelines(seeds)
        self.__fixed_seeds_path = sample_path
        self.__logger.info(f'Sampling {sample_n} seeds: success')

    def __classifier(self, c_type: ClassifierType):
        if c_type == ClassifierType.rfc:
            self.__logger.info('Classifying seeds with rfc...')
            rfc = RFCClassifier(self.__data_path, self.__fixed_seeds_path, self.__tool_path)
            c_seeds_path = rfc.classify()
            self.__logger.info('Classifying seeds with rfc: success')
            return c_seeds_path
        elif c_type == ClassifierType.rand:
            self.__logger.info('Classifying seeds with random sample...')
            n_sample = 10000
            rand = RandomizedSampleClassifier(self.__data_path, self.__fixed_seeds_path, self.__tool_path,
                                              n_sample=n_sample)
            r_seed_path = rand.classify()
            self.__logger.info(f'Classifying seeds with random sample {n_sample}: success')
            return r_seed_path
        elif c_type == ClassifierType.vec:
            self.__logger.info('Classifying seeds with ipv62vec...')
            ipv62vec = IPv62VecClassifier(self.__data_path, self.__fixed_seeds_path, self.__tool_path)
            v_seed_path = ipv62vec.classify()
            self.__logger.info('Classifying seeds with ipv62vec: success')
            return v_seed_path
        elif c_type == ClassifierType.eip:
            self.__logger.info('Classifying seeds with entropy-ip...')
            eip = EntropyIPClassifier(self.__data_path, self.__fixed_seeds_path, self.__tool_path)
            eip_seed_path = eip.classify()
            self.__logger.info('Classifying seeds with entropy-ip: success')
            return eip_seed_path
        else:
            pass

    @staticmethod
    def __addr62list(addr6: str) -> list:
        return [int(char, 16) * 17 for char in addr6.replace(':', '')]

    @staticmethod
    def __addr62bitmap(addr6: str) -> list:
        img = []
        addr_list = [int(char, 16) for char in addr6.replace(':', '')]
        for ny in addr_list:
            bitmap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            bitmap[ny * 2] = 255
            bitmap[ny * 2 + 1] = 255
            img.append(bitmap)
        return img

    def __image_gen_1(self, c_type: ClassifierType, seeds, save_path, _t):
        # 32个地址合成一张图
        img_list = []
        for index, line in enumerate(seeds):
            img_list += self.__addr62list(line.strip())
            if (index + 1) % 32 == 0:
                img_n = (index // 32)
                img = Image.new('L', (32, 32))
                for i in range(32):
                    for j in range(32):
                        p = i * 32 + j
                        img.putpixel((j, i), img_list[p])
                img_list = []
                img.save(os.path.join(save_path, f'{img_n:0>10}.png'))
                self.__logger.info(
                    f'type {c_type.value} | No.{img_n} img of {_t[:-4]} converted and saved at {save_path}')

    def __image_gen_2(self, c_type: ClassifierType, seeds, save_path, _t):
        # 一个地址合成一张图，简单重复32次
        for index, line in enumerate(seeds):
            img_list = []
            img_line = self.__addr62list(line.strip())
            for i in range(32):
                img_list.append(img_line)
            img = Image.fromarray(np.array(img_list, dtype=np.uint8), mode='L')
            img.save(os.path.join(save_path, f'{index:0>10}.png'))
            self.__logger.info(
                f'type {c_type.value} | No.{index} img of {_t[:-4]} converted and saved at {save_path}')

    def __image_gen_3(self, c_type: ClassifierType, seeds, save_path, _t):
        # 一个地址合成一张图，每一行都反转
        for index, line in enumerate(seeds):
            img_list = []
            img_line = self.__addr62list(line.strip())
            reverse = img_line[::-1]
            for i in range(32):
                if i % 2:
                    img_list.append(reverse)
                else:
                    img_list.append(img_line)
            img = Image.fromarray(np.array(img_list, dtype=np.uint8), mode='L')
            img.save(os.path.join(save_path, f'{index:0>10}.png'))
            self.__logger.info(
                f'type {c_type.value} | No.{index} img of {_t[:-4]} converted and saved at {save_path}')

    def __image_gen_4(self, c_type: ClassifierType, seeds, save_path, _t):
        # 一个地址合成一张图,每一列标识一个nyyble，标识办法是 每两行作为一个标记位，01行标记nyyble=0，12行标记nybble=1...
        for index, line in enumerate(seeds):
            img = self.__addr62bitmap(line.strip())
            img = Image.fromarray(np.array(img, dtype=np.uint8).T, mode='L')
            img.save(os.path.join(save_path, f'{index:0>10}.png'))
            self.__logger.info(
                f'type {c_type.value} | No.{index} img of {_t[:-4]} converted and saved at {save_path}')

    def convert2img(self, c_type: ClassifierType, gen_type=2, sample_n=10000):
        """
        Convert data to images
        :param c_type: classifier type
        :param gen_type: 图生成方法选择
        :param sample_n: 采样数，也即数量上限
        :return: None
        """
        self.__logger.info('Convert seeds to images')
        c_seeds_path = self.__classifier(c_type)
        #c_seeds_path = './resources/data/classify_data/rfc'
        self.__logger.info('Converting classified seeds to images...')
        for _t in os.listdir(c_seeds_path):
            if _t.endswith('.txt'):
                save_path = os.path.join(self.__img_path, c_type.value, _t[:-4], 'src')
                check_del_path(save_path)
                check_fix_path(save_path)
                with open(os.path.join(c_seeds_path, _t), 'r') as f:
                    seeds = f.readlines()
                    if len(seeds) > sample_n:
                        self.__logger.debug(f'{_t[:-4]} seeds size of {len(seeds)}, sampling {sample_n} seeds')
                        seeds = random.sample(seeds, sample_n)
                        self.__logger.debug('sample DONE!')
                    if gen_type == 1:
                        self.__image_gen_1(c_type, seeds, save_path, _t)
                    elif gen_type == 2:
                        self.__image_gen_2(c_type, seeds, save_path, _t)
                    elif gen_type == 3:
                        self.__image_gen_3(c_type, seeds, save_path, _t)
                    elif gen_type == 4:
                        self.__image_gen_4(c_type, seeds, save_path, _t)
                    else:
                        raise ValueError('gen_type not found')
        self.__logger.info('Converting classified seeds to images: success')


if __name__ == '__main__':
    dp = DataProcessor()
    dp.sample_source(sample_n=300000)
    dp.convert2img(ClassifierType.eip, gen_type=4,sample_n=10000)
