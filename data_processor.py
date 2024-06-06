from utils.classify_type import ClassifierType
import os
import warnings
from PIL import Image
from logger import TgaLogger


def check_fix_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


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
        self.temp = os.path.join(self.data_path, 'temp')
        check_fix_path(self.out_path)
        check_fix_path(self.temp)

    def classify(self):
        pass


class RFCClassifier(Classifier):

    def __init__(self, data_path, seeds_path, tool_path, *, log_name='rfc_classifier'):
        super().__init__(data_path, seeds_path, tool_path, log_name=log_name)
        self.__logger = self.logger
        self.__out_path = os.path.join(self.out_path, 'rfc')
        check_fix_path(self.__out_path)
        self.__rfc_profile = os.path.join(self.temp, 'rfc_profile.txt')
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
        self.__tool_path = os.path.join(self.__root_path, 'tools')
        self.__resource_path = os.path.join(self.__root_path, 'resources')
        self.__data_path = os.path.join(self.__resource_path, 'data')
        self.__img_path = os.path.join(self.__resource_path, 'img')
        self.__alias_path = os.path.join(self.__resource_path, 'alias')
        for p in self.__root_path, self.__resource_path, self.__data_path, self.__img_path, self.__alias_path:
            check_fix_path(p)
        self.__seeds_path = os.path.join(self.__data_path, 'seeds.txt')
        self.__fixed_seeds_path = os.path.join(self.__data_path, 'fixed_seeds.txt')
        self.__alias_path = os.path.join(self.__alias_path, 'aliased-prefixes.txt')
        if not os.path.exists(self.__seeds_path) or not os.path.exists(self.__alias_path):
            warnings.warn('No seeds or alias found, please check the data folder')
        if not os.path.exists(self.__fixed_seeds_path):
            # 转为ipv6地址完全形式
            os.system(f'cat "{self.__seeds_path}" | "{self.__tool_path}/addr6" -i -f > "{self.__fixed_seeds_path}"')

    def __classifier(self, c_type: ClassifierType):
        if c_type == ClassifierType.rfc:
            self.__logger.info('Classifying seeds with rfc...')
            rfc = RFCClassifier(self.__data_path, self.__fixed_seeds_path, self.__tool_path)
            c_seeds_path = rfc.classify()
            self.__logger.info('Classifying seeds with rfc: success')
            return c_seeds_path
        elif c_type == 'other':
            pass
        else:
            pass

    @staticmethod
    def __addr62list(addr6: str) -> list:
        return [pow(int(char, 16), 2) for char in addr6.replace(':', '')]

    def convert2img(self, c_type: ClassifierType):
        """
        Convert data to images
        :param c_type: classifier type
        :return: None
        """
        self.__logger.info('Convert seeds to images')
        c_seeds_path = self.__classifier(c_type)
        self.__logger.info('Converting classified seeds to images...')
        for _t in os.listdir(c_seeds_path):
            if _t.endswith('.txt'):
                save_path = os.path.join(self.__img_path, c_type.value, _t[:-4], 'src')
                check_fix_path(save_path)
                with open(os.path.join(c_seeds_path, _t), 'r') as f:
                    img_list = []
                    for index, line in enumerate(f):
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
                        # 舍弃最后数量不足32的地址
        self.__logger.info('Converting classified seeds to images: success')


if __name__ == '__main__':
    dp = DataProcessor()
    dp.convert2img(ClassifierType.rfc)
