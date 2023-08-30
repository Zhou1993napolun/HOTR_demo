import glob
import os
from pathlib import Path

import cv2
import numpy as np
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
vid_formats = ['.mov', '.avi', '.mp4']
class LoadImages:  # for inference
    def __init__(self, path, img_size=416, half=False):
        """
        进行推理的图像预处理
        :param path: 需要检测的图片文件夹 'data/samples'
        :param img_size: 416
        :param half: 是否采用半精度推理 False
        """
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            # files：列表，包含了待检测的图片路径[图片1，图片2...]
            # 'data\\samples\\bus.jpg'
            # 'data\\samples\\zidane.jpg'
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        # os.path.splitext(“文件路径”)    分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
        # os.path.splitext(x): ('data\\samples\\bus', '.jpg')
        # os.path.splitext(x)[-1].lower(): '.jpg'
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats] # 判断是否是支持的图片格式
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats] # 判断是否是支持的视频格式
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files 总共要检测的数目
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        self.half = half  # half precision fp16 images
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # 如果迭代次数等于图片数目，就停止迭代
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count] # 得到第self.count张图片路径

        if self.video_flag[self.count]: # 如果有视频的话
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # 迭代次数加一
            self.count += 1
            # Read image 读取图片
            img0 = cv2.imread(path)  # BGR HWC: (1080, 810, 3)
            assert img0 is not None, 'Image Not Found ' + path
            # image 1/2 data/samples/bus.jpg:
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        # img, *_ = letterbox(img0, new_shape=self.img_size) # img经过padding后的最小输入矩形图: (416, 320, 3)
        img = img0
        # cv2.imshow('Padded Image', img)
        # cv2.waitKey()

        # Normalize RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB  HWC2CHW: (3, 416, 320)
        # # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        # img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap
    
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416, half=False, pipe = 'http://pi:raspberry@192.168.12.150:8090/stream.mjpg'):
        self.img_size = img_size
        self.half = half  # half precision fp16 images

        pipe = pipe  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.cap = cv2.VideoCapture(pipe)  # video capture object

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cap.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        # img, *_ = letterbox(img0, new_shape=self.img_size)
        img=img0

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, img, img0, None

    def __len__(self):
        return 0
