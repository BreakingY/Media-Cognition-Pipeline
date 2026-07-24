# -*-coding:utf-8 -*-
import mcp_module
import time
import cv2
import numpy as np
import threading

objects = [None, None, None, None]

inputs = [
    "../Test/test1.mp4",
    "../Test/test1.mp4",
    "../Test/test1.mp4",
    "../Test/test1.mp4",
]

outputs = [
    "./output1.mp4",
    "./output2.mp4",
    "./output3.mp4",
    "./output4.mp4",
]

test_cnt = 4

class FrameRateStats:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.frame_count = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.lock = threading.Lock()

    def on_frame(self):
        with self.lock:
            self.frame_count += 1

    def get_and_reset_recent_frames(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_print_time

            fps = self.frame_count / elapsed if elapsed > 0 else 0

            self.frame_count = 0
            self.last_print_time = now

            return fps


stats = [FrameRateStats(i) for i in range(test_cnt)]


def create_video_callback(stream_id):
    def on_image(img):
        stats[stream_id].on_frame()

        mcp_module.addVideoFrame(objects[stream_id], img)

    return on_image

def create_audio_callback(stream_id):
    def on_pcm(pcm, data_len, spb, channels):
        """
        pcm: numpy uint8 array packed PCM
        data_len: 每通道sample数量
        spb: 每个sample字节数
        channels: 通道数量
        """

        mcp_module.addAudioFrame(objects[stream_id], pcm, data_len, spb, channels)

    return on_pcm


video_callbacks = [create_video_callback(i)
    for i in range(test_cnt)
]

audio_callbacks = [
    create_audio_callback(i)
    for i in range(test_cnt)
]


def print_frame_rates():
    while True:
        time.sleep(2)
        print("\n" + "=" * 60)
        for i in range(test_cnt):
            fps = stats[i].get_and_reset_recent_frames()
            print(f"stream {i}: {fps:.2f} FPS")
        print("=" * 60)


if __name__ == "__main__":
    mcp_module.initialization()

    t = threading.Thread(target=print_frame_rates, daemon=True)
    t.start()

    for i in range(test_cnt):
        print(f"create stream {i}")

        objects[i] = mcp_module.createMcpObject(inputs[i], outputs[i], 0)

        mcp_module.registerImgcallback(objects[i], video_callbacks[i], audio_callbacks[i])

    print("all streams started")
    del_cnt = 0
    while del_cnt != test_cnt:
        for i in range(test_cnt):
            if objects[i] and mcp_module.overHandle(objects[i]) is True:
                print(f"destroy stream {i}")
                mcp_module.destoryMcpObject(objects[i],)
                objects[i] = None
                del_cnt += 1
        time.sleep(1)
        
    mcp_module.cleanUp()
    print("done")