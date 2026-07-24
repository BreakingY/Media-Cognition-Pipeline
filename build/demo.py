# -*-coding:utf-8 -*-
import mcp_module
import time
import cv2
import numpy as np
obj = None
def on_image(img):
    h, w, c = img.shape

    cv2.rectangle(img, (0, 0), (50, 50), (0, 0, 255), 2)

    mcp_module.addVideoFrame(obj, img)
def on_pcm(pcm, data_len, spb, channels):
    '''
    pcm:packed模式pcm数据
    data_len:单通道当本个数
    spb:每个样本占用的字节数
    channels:通道数
    '''
    mcp_module.addAudioFrame(obj, pcm, data_len, spb, channels)

if __name__ == "__main__":
    
    mcp_module.initialization()

    obj = mcp_module.createMcpObject("../Test/test1.mp4", "./output.mp4", 0)
    mcp_module.registerImgcallback(obj, on_image, on_pcm)
    while mcp_module.overHandle(obj) is False:
        time.sleep(1)
    mcp_module.destoryMcpObject(obj)

    mcp_module.cleanUp()