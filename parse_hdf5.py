# -*- coding: utf-8 -*-
import h5py
import json
import struct
import numpy as np
import pandas as pd


class Hdf5Parse(object):
    """
    这个类主要是读取HDF5文件并根据header里面的configuration解析各种传感器数据，包含header, GPS，CAN，IMU数据, 并返回各种解析之后的数据
    """

    def __init__(self, hdf5_file):
        """
        读取hdf5文件，获取header文件，并判断header做异常处理
        :param hdf5_file: 包含原始二进制传感器数据流
        """
        self.file = hdf5_file
        self.header_flg = "header"
        self.hdf5_data_header = None
        self.config = "configurations"
        self.channels = "channels"
        # 存储各个信号的数据解析格式, eg:{'format': {'gps': 'fff', 'can': 'ffffffff'}, 'label': {'gps': [], 'imu': [], 'can': []}}
        self.configuration_dict = {}
        # # 存储各个信号数据    eg: {'imu': numpy_array, 'gps': numpy_array, 'can': numpy_array}
        self.dataset_data_dict = {}
        # # 存储各个信号数据    eg: {'imu': Dataframe, 'gps': Dataframe, 'can': Dataframe}
        self.dataset_dataFrame_dict = {}
        # 存储所有信息， eg: {'header': {}, 'data': {'imu': numpy_array, 'gps': numpy_array, 'can': numpy_array}}
        self.hdf5_dict = {}
        # 读取h5文件并提取数据
        self.run()

    def run(self):
        """
        主程序读取hdf5文件并且判断并提取解析格式并生成解析好的数据
        :return:
        """
        # logger.info("Hdf5Parse run -------------------")
        # ## 检查是否含有header
        self.check_hdf5_header()

    def get_gdf5_data(self):
        """
        外部类调取parse类生成的数据
        :return:
        """
        if self.hdf5_dict:
            return self.hdf5_dict

    def get_h5Data_dataFrame(self):
        """
        :return:
        """
        if self.dataset_dataFrame_dict:
            return self.dataset_dataFrame_dict

    def check_hdf5_header(self):
        """
        对源文件hdf5进行check处理
        :return:
        """
        # logger.debug("at {0}".format(sys._getframe().f_code.co_name))
        with h5py.File(self.file, 'r') as f:
            if self.header_flg in f.keys():
                hdf5_data_header = f[self.header_flg][()]  # 获取header数据信息
                self.hdf5_data_header = json.loads(hdf5_data_header)  # 接收header数据
                if self.hdf5_data_header:
                    self.check_hdf5_config()  # 检查configuration
                    self.hdf5_dict['header'] = self.hdf5_data_header
                else:
                    raise ValueError("不能解析HDF5文件，header为空！！！！！！")
            else:
                raise ValueError("不能解析HDF5文件，没有header文件！！！！！！")
            #  解析保存原始数据
            for h5_dataset in f.keys():
                # print('h5文件的key:', h5_dataset)
                if h5_dataset != self.header_flg:
                    hdf5_data_value = f[h5_dataset][()]  # 获取gps, path, imu, can数据

                    h5dataType = type(hdf5_data_value)
                    # print('数据的格式:', h5dataType)
                    if h5dataType == np.ndarray:  # numpy 数组存储数据
                        self.dataset_data_dict[h5_dataset] = hdf5_data_value
                        # convert numpy to pandas
                        label_dict = self.configuration_dict['label']
                        key = h5_dataset + 'Configuration'
                        if key in label_dict.keys():
                            self.dataset_dataFrame_dict[h5_dataset] = pd.DataFrame(data=hdf5_data_value, columns=label_dict[key])
                    elif h5dataType == np.void:  # 二进制流存储数据
                        self.parse_h5_data(h5_dataset, hdf5_data_value)
                    else:
                        print("H5 数据类型{0}不存在 ".format(h5dataType))
            self.hdf5_dict['data'] = self.dataset_data_dict

    def check_hdf5_config(self):
        """
        提取解析的configuration的format
        :return:
        """
        # logger.debug("at {0}".format(sys._getframe().f_code.co_name))
        if self.config not in self.hdf5_data_header.keys():
            raise ValueError("不能解析HDF5文件，configuration不存在！！！！！！")
        configurations = self.hdf5_data_header[self.config]  # 获取configurations里面字典信息
        formatDict = {}
        labelDict = {}
        for conf_name, cond_dict in configurations.items():  # conf_name:pathConfiguration cond_dict:{"channels":[{"name
            fmt = ""
            label = []
            if self.channels not in cond_dict:
                raise ValueError("不能解析HDF5文件，channels不存在！！！！！！")
            for each_channel in cond_dict[self.channels]:  # gps imu的信号list
                typ = each_channel['type']  # 获取数据对应的数据类型
                name = each_channel['name']  # 获取数据对应的数据名称
                label.append(name)
                if typ == 'float32':
                    fmt += 'f'
                elif typ == 'float':
                    fmt += 'f'
                elif typ == 'uint32':
                    fmt += 'I'
                elif typ == 'int32':
                    fmt += 'i'
                elif typ == 'int16':
                    fmt += 'h'
                elif typ == 'uint16':
                    fmt += 'H'
                elif typ == 'double':
                    fmt += 'd'
                elif typ == 'int64':
                    fmt += 'q'
                elif typ == 'uint64':
                    fmt += 'Q'
                elif typ == 'char':
                    fmt += 'c'
                elif typ == 'uchar':
                    fmt += 'B'
                elif typ == 'uint8':
                    fmt += 'B'
            formatDict[conf_name] = fmt  # 数据类型
            labelDict[conf_name] = label  # 数据名
        self.configuration_dict['format'] = formatDict
        self.configuration_dict['label'] = labelDict
        # print('self.configuration_dict', self.configuration_dict)

    def parse_h5_data(self, dataset_name, data):
        """
        处理h5文件中的二进制数据，根据config提取的相应的fmt解析h5文件中的数据dataset并输出
        :return:
        """
        count = int(self.hdf5_data_header['recordCount'])
        fmort_dict = self.configuration_dict['format']
        label_dict = self.configuration_dict['label']
        for config_name in fmort_dict.keys():
            if dataset_name in config_name:
                fmt = fmort_dict[config_name]
                field_len = len(fmt)
                fmt = '<' + fmt * count
                gen_data = struct.unpack(fmt, data)
                output_data = np.zeros((count, field_len))
                for i in range(field_len):
                    output_data[:, i] = list(gen_data[i::field_len])
                self.dataset_data_dict[dataset_name] = output_data
                self.dataset_dataFrame_dict[dataset_name] = pd.DataFrame(data=output_data, columns=label_dict[config_name])


