import enum


class ClassifierType(enum.Enum):
    rfc = 'rfc'
    vec = 'ipv62vec'
    rand = 'random'
    eip = 'entropy-ip'
