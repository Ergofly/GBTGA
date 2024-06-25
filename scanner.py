import os
import csv
import subprocess
import schedule
import time
from datetime import datetime


class Ipv6Scanner:
    def __init__(self, address_path, res_path):
        self.__address_path = address_path
        self.__res_path = res_path
        self.__ping_res = os.path.join(self.__res_path, 'ping_res.csv')
        self.__tcp_synscan_80_res = os.path.join(self.__res_path, 'tcp_synscan_80_res.csv')
        self.__tcp_synscan_443_res = os.path.join(self.__res_path, 'tcp_synscan_443_res.csv')
        self.__tcp_synopt_80_res = os.path.join(self.__res_path, 'tcp_synopt_80_res.csv')
        self.__tcp_synopt_443_res = os.path.join(self.__res_path, 'tcp_synopt_443_res.csv')
        self.__cmd_icmp = f'sudo zmap --ipv6-source-ip=240d:c000:f000:9c00:0:9c48:c1c6:a3cd --ipv6-target-file={self.__address_path} -M icmp6_echoscan -B 5M -o {self.__ping_res}'
        self.__cmd_tcp_synscan_80 = f'sudo zmap --ipv6-source-ip=240d:c000:f000:9c00:0:9c48:c1c6:a3cd --ipv6-target-file={self.__address_path} -M ipv6_tcp_synscan -B 5M -o {self.__tcp_synscan_80_res} -p 80'
        self.__cmd_tcp_synscan_443 = f'sudo zmap --ipv6-source-ip=240d:c000:f000:9c00:0:9c48:c1c6:a3cd --ipv6-target-file={self.__address_path} -M ipv6_tcp_synscan -B 5M -o{self.__tcp_synscan_443_res} -p 443'
        self.__cmd_tcp_synopt_80 = f'sudo zmap --ipv6-source-ip=240d:c000:f000:9c00:0:9c48:c1c6:a3cd --ipv6-target-file={self.__address_path} -M ipv6_tcp_synopt -B 5M -o {self.__tcp_synopt_80_res} -p 80'
        self.__cmd_tcp_synopt_443 = f'sudo zmap --ipv6-source-ip=240d:c000:f000:9c00:0:9c48:c1c6:a3cd --ipv6-target-file={self.__address_path} -M ipv6_tcp_synopt -B 5M -o {self.__tcp_synopt_443_res} -p 443'
        self.__last_result = None

    def __scan(self):
        commands = [self.__cmd_icmp, self.__cmd_tcp_synscan_80, self.__cmd_tcp_synscan_443, self.__cmd_tcp_synopt_80,
                    self.__cmd_tcp_synopt_443]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f'Command {" ".join(command)} failed with return code {result.returncode}')
                print(result.stderr)
                continue
        files = [self.__ping_res, self.__tcp_synscan_80_res, self.__tcp_synscan_443_res, self.__tcp_synopt_80_res,
                 self.__tcp_synopt_443_res]

        address_sets = []
        for f in files:
            with open(f, mode='r', newline='') as file:
                reader = csv.reader(file)
                address_set = {row[0] for row in reader}  # 将第一列转为集合
                address_sets.append(address_set)

        # 取并集
        all_addresses = set.union(*address_sets)

        # 如果之前有生成的txt文件，取其内容与新地址的交集
        if self.__last_result:
            with open(self.__last_result, 'r') as f:
                old_addresses = {line.strip() for line in f}
            common_addresses = all_addresses.union(old_addresses)
        else:
            common_addresses = all_addresses

        # 写入到新的txt文件
        # 获取当前时间
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        # 定义输出文件名
        output_file_name = f'{current_time}_{len(common_addresses)}.txt'
        output_file = os.path.join(self.__res_path, output_file_name)

        with open(output_file, 'w') as f:
            for address in common_addresses:
                f.write(f'{address}\n')

        # 更新最新的地址文件
        self.__last_result = output_file
        print(f'Hit addresses written to {output_file}')

    def scan(self):
        schedule.every().hour.do(self.__scan)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == '__main__':
    sc = Ipv6Scanner('addresses.txt', './res')
    sc.scan()
