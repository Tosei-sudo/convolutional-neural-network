# coding: utf-8

import struct
import numpy as np

class File:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.eof = False
    
    def read(self, size):
        self.pos += size
        self.eof = self.pos >= len(self.data)
        return self.data[self.pos-size:self.pos]
    
    def next(self, size = 1, offset = 0):
        tmp_pos = self.pos + size + offset
        return self.data[tmp_pos-size:tmp_pos]

class BMPHeader:
    def read(self, file):
        self.signature = file.read(2)
        self.file_size = struct.unpack('<I', file.read(4))[0]
        self.reserved1 = struct.unpack('<H', file.read(2))[0]
        self.reserved2 = struct.unpack('<H', file.read(2))[0]
        
        self.offset = struct.unpack('<I', file.read(4))[0]

class BMPInfoHeader:
    def read(self, file):
        self.size = struct.unpack('<I', file.read(4))[0]
        self.width = struct.unpack('<I', file.read(4))[0]
        self.height = struct.unpack('<I', file.read(4))[0]
        self.planes = struct.unpack('<H', file.read(2))[0]
        self.bits_per_pixel = struct.unpack('<H', file.read(2))[0]
        self.compression = struct.unpack('<I', file.read(4))[0]
        self.image_size = struct.unpack('<I', file.read(4))[0]
        self.x_pixels_per_meter = struct.unpack('<I', file.read(4))[0]
        self.y_pixels_per_meter = struct.unpack('<I', file.read(4))[0]
        self.colors_used = struct.unpack('<I', file.read(4))[0]
        self.colors_important = struct.unpack('<I', file.read(4))[0]

class BMP:
    def read_uncompressed(self, file):
        data_size = self.infoHeader.height * self.infoHeader.width * 3
        raw_data = file.read(data_size)
        self.data = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.infoHeader.height, self.infoHeader.width, 3))
    
    def read(self, file):
        self.header = BMPHeader()
        self.header.read(file)
        
        self.infoHeader = BMPInfoHeader()
        self.infoHeader.read(file)

        if self.infoHeader.compression == 0:
            self.read_uncompressed(file)