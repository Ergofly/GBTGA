from data_process import DataProcessor
from train import Trainner
from gen import IPv6Generator
from utils.classify_type import ClassifierType

type = ClassifierType.rfc

# dp = DataProcessor()
# dp.new_convert2img(type)
t = Trainner()
t.train(type)
gen = IPv6Generator()
gen.gen(type)
