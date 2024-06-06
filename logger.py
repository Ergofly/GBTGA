import os
import logging
from logging.handlers import RotatingFileHandler


class ColorHandler(logging.StreamHandler):
    GRAY8 = "38;5;8"
    GRAY7 = "38;5;7"
    ORANGE = "33"
    RED = "31"
    WHITE = "0"
    PURPLE = "35"
    BLUE = "34"

    def emit(self, record):
        try:
            msg = self.format(record)
            level_color_map = {
                logging.DEBUG: self.PURPLE,
                logging.INFO: self.BLUE,
                logging.WARNING: self.ORANGE,
                logging.ERROR: self.RED,
                logging.CRITICAL: self.GRAY8

            }

            csi = f"{chr(27)}["  # control sequence introducer
            color = level_color_map.get(record.levelno, self.WHITE)

            self.stream.write(f"{csi}{color}m{msg}{csi}m\n")
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class TgaLogger:
    """
    日志模块
    """
    __logger = None
    __log_path = os.path.join(os.path.curdir, 'logs')
    __max_file_size = 2 * 1024 * 1024  # 设置为2MB
    __backup_count = 5  # 保留最近5个备份文件

    def __init__(self, logger_name, log_path=None):
        if log_path is not None:
            self.__log_path = os.path.realpath(log_path)
        if not os.path.exists(self.__log_path):
            os.mkdir(self.__log_path)
        self.__setlog(logger_name)

    def __setlog(self, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(logging.DEBUG)

        ch = ColorHandler()
        ch.setLevel(logging.DEBUG)

        fh = RotatingFileHandler(os.path.join(self.__log_path, logger_name + '.log'), maxBytes=self.__max_file_size,
                                 backupCount=self.__backup_count, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.__logger.addHandler(ch)
        self.__logger.addHandler(fh)

    def get_logger(self):
        return self.__logger


if __name__ == '__main__':
    tlog = TgaLogger('test').get_logger()
    tlog.debug('test')
    tlog.critical('test')
    tlog.info('test')
    tlog.warning('test')
    tlog.error('test')
