#! /usr/local/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pickle

from PIL import Image

histSize = 64
imageSize = (256, 256) 

def SplitImg(img, x, y):
    h, w = img.shape[:2]
    split_imgs = []
    cx = 0
    cy = 0

    split_w = w//x
    split_h = h//y

    for i in range(x):
        for j in range(y):
            split_imgs.append(img[cy:cy+split_h,cx:cx+split_w,:])
            cy+=h//y
        cy=0
        cx+=split_w

    return np.array(split_imgs)




def ColorHist(img_path, color_type, split_num=1):
    img = cv2.imread(img_path)
    hists = []
    if color_type == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_type == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_type == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    
    imgs = SplitImg(img, split_num, split_num)
    
    for img in imgs:
        ch0, ch1, ch2 = img[:,:,0], img[:,:,1], img[:,:,2]
        img_shape = img.shape[0]*img.shape[1]
        
        hist_ch0 = cv2.calcHist([ch0],[0],None,[histSize],[0,256])/img_shape
        hist_ch1 = cv2.calcHist([ch1],[0],None,[histSize],[0,256])/img_shape
        hist_ch2 = cv2.calcHist([ch2],[0],None,[histSize],[0,256])/img_shape
        
        hists.append(np.array([hist_ch0, hist_ch1, hist_ch2])/3)
    return np.array(hists)[:,:,:,0]/(split_num*split_num)


def DCNNF(img_path):
    vgg16 = models.vgg16(pretrained=True,progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_transform = transforms.Compose([transforms.Resize(imageSize),transforms.ToTensor(),normalize])
    vgg16fc7 = torch.nn.Sequential(
        vgg16.features,
        vgg16.avgpool,
        nn.Flatten(),
        *list(vgg16.classifier.children())[:-3]
    )
    
    img = Image.open(img_path)
    tonsor_img = image_transform(img)
    tonsor_img = tonsor_img.unsqueeze(0)
    vgg16fc7.eval()
    with torch.no_grad():
        fc7=vgg16fc7(tonsor_img).numpy()
    return fc7[0]/np.sum(fc7[0])

def Features(img_paths):
    rgb_hists1x1 = []
    rgb_hists2x2 = []
    rgb_hists3x3 = []
    hsv_hists1x1 = []
    hsv_hists2x2 = []
    hsv_hists3x3 = []
    luv_hists1x1 = []
    luv_hists2x2 = []
    luv_hists3x3 = []
    dcnnf = []
    for img_path in img_paths:
        rgb_hists1x1.append(ColorHist(img_path, 'RGB', 1))
        rgb_hists2x2.append(ColorHist(img_path, 'RGB', 2))
        rgb_hists3x3.append(ColorHist(img_path, 'RGB', 3))
        hsv_hists1x1.append(ColorHist(img_path, 'HSV', 1))
        hsv_hists2x2.append(ColorHist(img_path, 'HSV', 2))
        hsv_hists3x3.append(ColorHist(img_path, 'HSV', 3))
        luv_hists1x1.append(ColorHist(img_path, 'LUV', 1))
        luv_hists2x2.append(ColorHist(img_path, 'LUV', 2))
        luv_hists3x3.append(ColorHist(img_path, 'LUV', 3))
        dcnnf.append(DCNNF(img_path))
    
    rgb_hists1x1 = np.array(rgb_hists1x1)
    rgb_hists2x2 = np.array(rgb_hists2x2)
    rgb_hists3x3 = np.array(rgb_hists3x3)
    hsv_hists1x1 = np.array(hsv_hists1x1)
    hsv_hists2x2 = np.array(hsv_hists2x2)
    hsv_hists3x3 = np.array(hsv_hists3x3)
    luv_hists1x1 = np.array(luv_hists1x1)
    luv_hists2x2 = np.array(luv_hists2x2)
    luv_hists3x3 = np.array(luv_hists3x3)
    dcnnf = np.array(dcnnf)
    
    np.save('data/rgb_hists1x1', rgb_hists1x1)
    np.save('data/rgb_hists2x2', rgb_hists2x2)
    np.save('data/rgb_hists3x3', rgb_hists3x3)
    np.save('data/hsv_hists1x1', hsv_hists1x1)
    np.save('data/hsv_hists2x2', hsv_hists2x2)
    np.save('data/hsv_hists3x3', hsv_hists3x3)
    np.save('data/luv_hists1x1', luv_hists1x1)
    np.save('data/luv_hists2x2', luv_hists2x2)
    np.save('data/luv_hists3x3', luv_hists3x3)
    np.save('data/dcnnf', dcnnf)
if __name__ == '__main__':
    imglist=glob.glob('imgdata/okapi_img/*.jpg')+glob.glob('imgdata/zebra_img/*.jpg')

    features = Features(imglist)
    f = open('data/img_path.txt', 'wb')
    pickle.dump(imglist, f)