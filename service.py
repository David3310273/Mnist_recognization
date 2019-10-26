import torch
import torchvision.transforms as transforms
import json
import numpy as np
import cv2
from lenet5 import lenet5
from PIL import Image
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import cgi
import logging
import io


class LenetHandler(BaseHTTPRequestHandler):
    """
    请求类
    """
    __width, __height = 28, 28  # 模型输入尺寸

    def __predict(self, img, model="./model/lenet5"):
        """
        给定一张图片输入模型，预测其数字是多少
        :param img: 输入图片的numpy数组
        :param model: 模型路径
        :return:
        """

        # TODO：单例模式加载模型
        mnist = lenet5()
        mnist.load_state_dict(torch.load(model))
        mnist.eval()    # 注意添加这行语句，使得dropout和一些随机性的网络层去除随机特性

        input_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        image = input_transform(img)
        result = mnist(torch.unsqueeze(image, dim=0))

        return torch.argmax(result).item()

    def _set_headers(self, statusCode=200):
        self.send_response(statusCode)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        """
        解析HTTP请求中上传的图片
        :return:
        """
        ctype, pdict = cgi.parse_header(self.headers.get("Content-type"))
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")   # python3兼容byte格式

        # 如果是其他类型参数，返回400
        if ctype == "multipart/form-data":
            postvars = cgi.parse_multipart(self.rfile, pdict)
            pic_nums = []

            for key, val in postvars.items():
                image = Image.open(io.BytesIO(val[0]))  # 使用pillow读取请求中的字节流
                result = self.__predict(image)
                pic_nums.append(result)

            self._set_headers()
            self.wfile.write(json.dumps({
                "result": pic_nums
            }).encode())
        else:
            self._set_headers(400)

    def do_GET(self):
        """
        禁止GET请求
        :return:
        """
        self._set_headers(403)

class LenetServer(HTTPServer):
    """
    服务器类
    """
    def __init__(self, handler_class=BaseHTTPRequestHandler, host="localhost", port=80):
        self.__address = (host, port)
        self.__handler = handler_class
        super().__init__(self.__address, handler_class)

    def run(self):
        self.serve_forever()

if __name__ == "__main__":
    server = LenetServer(LenetHandler, port=8081)
    server.run()
