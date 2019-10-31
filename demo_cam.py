from __future__ import division
import pyrealsense2 as rs
from models import *
from utils.utils import *
from utils.datasets import *
import cv2

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


parser = argparse.ArgumentParser()
parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
opt = parser.parse_args()
print(opt)

classes = ['274078.4-wechselschloss', '343302.0-ritzelwelle', '370054.0-schwungscheibe', "379712.1-nockenwelle", "391391.1-hammer"]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

print('loading', opt.weights_path)
if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode

print("loading classes")
classes_ = load_classes(opt.class_path)  # Extracts class labels from file

#Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def get_model_input(img, input_dim):
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float().cuda()
    img_ = Variable(img_)

    return img_



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
print("starting realsense pipeline")
pipeline.start(config)
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        
        # Convert images to numpy arrays
        im0 = np.asanyarray(color_frame.get_data())
        img = get_model_input(im0, 416)
        pred = model(img)
        # Run NMS
        pred = non_max_suppression(pred, conf_thres=0.96, nms_thres=0.4)

        for i, det in enumerate(pred):
            s = '%g: ' % i
            for di, d in enumerate(pred):
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, _, cls in det:
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        print(label)

        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        if(pred):
            cv2.imshow('RealSense', im0)
        key = cv2.waitKey(1)
        if key == ord("c"):
            print("run inference")
            
        


        if key in (27, ord("q")):
            break

finally:

    # Stop streaming
    pipeline.stop()    
