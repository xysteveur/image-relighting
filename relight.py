'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

from face_detect.faceDetect import cropFace, rdmCrop

# other modules
import os
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import argparse

# This code is adapted from https://github.com/zhhoper/DPR

def parse_args():
    parser = argparse.ArgumentParser(
        description="image relighting training.")
    parser.add_argument(
        '--source_image',
        default='obama.jpg',
        help='name of image stored in data/',
    )
    parser.add_argument(
        '--light_image',
        default='obama.jpg',
        help='name of image stored in data/',
    )
    parser.add_argument(
        '--model',
        default='trained.pt',
        help='model file to use stored in trained_model/'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='cpu vs. gpu'
    )
    parser.add_argument(
        '--face_detect',
        default='Neither',
        help='Options: "both" or "light". Face detection/cropping for more accurate relighting.'
    )

    parser.add_argument(
        '--crop_paste',
        action='store_true',
        help='Options: apply local lighting adjustment'
    )
    
    

    return parser.parse_args()

def preprocess_image(img_path, srcOrLight):
    src_img = cv2.imread(img_path)
    if (ARGS.face_detect == 'both') or (ARGS.face_detect == 'light' and srcOrLight == 2):
        # src_img = cropFace(src_img)
        # src_img = rdmCrop(src_img, Light=WLight)
        pass
    row, col, _ = src_img.shape
    img_ori = src_img[:,:,:]
    src_img = cv2.resize(src_img, (256, 256))
    Lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...] #not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL))
    if (ARGS.gpu):
        inputL = inputL.cuda()
    return inputL, row, col, Lab, img_ori

def generate_mask(img, img_name, index, th=500, tw=300):

    pixels = img
    h, w = len(pixels), len(pixels[0])
    print(f"generate mask ...")

    # th, tw = torch.randint(0,h//2, size=(1,)).item(), torch.randint(0,w//2, size=(1,)).item()

    if h < th or w < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    y = torch.randint(10, (h - th + 1) // 2, size=(1,)).item()
    x = torch.randint(50, (w - tw + 1) // 2 - 200, size=(1,)).item()

    y_ = torch.randint((h - th + 1) // 2, h - th + 1-100, size=(1,)).item()
    x_ = torch.randint((w - tw + 1) // 2 + 200, w - tw + 1-100, size=(1,)).item()
  
    x2, y2 = x + tw, y + th
    x2_, y2_ = x_ + tw, y_ + th
    print(f"x:{(x,x2)}, y:{(y,y2)}, tw:{(x_,x2_)}, th:{(y_,y2_)}")

    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if (x<j<x2 and y<i<y2) or (x_<j<x2_ and y_<i<y2_):
                mask[i,j] = 1.0
            

    # Creating the kernel(2d convolution matrix) 
    kernel1 = np.ones((5, 5), np.float32)/25
  
    # Applying the filter2D() function 
    mask = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel1) 
    print(np.max(mask))
    print(np.min(mask))
    mask_out = (mask*255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(saveFolder,
         '{}_{:02d}_mask.jpg'.format(img_name,index)), mask_out)

    return mask

def apply_mask(Limg, img, mask):
    # kernel = np.ones((5,5),np.float32)/25
    mask = mask[:,:,np.newaxis]
    img_blur = Limg * mask + (1-mask)*img
    kernel = np.ones((5, 5), np.float32)/25
    img_blur = cv2.filter2D(img_blur,-1,kernel)
    
    return img_blur

ARGS = parse_args()

modelFolder = 'trained_models/'

# load model
from model import *
my_network = HourglassNet()

if (ARGS.gpu):
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model)))
    my_network.cuda()
else:
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model), map_location=torch.device('cpu')))

my_network.train(False)

saveFolder = 'result_both_crop_light_0124_light1'
saveFolder = os.path.join(saveFolder, ARGS.model.split(".")[0])
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

light_img, _, _, _,_ = preprocess_image('data/test/images/{}'.format(ARGS.light_image), 2)

sh = torch.zeros((1,9,1,1))
if (ARGS.gpu):
    sh = sh.cuda()

_, outputSH  = my_network(light_img, sh, 0)

src_img, row, col, Lab, img_ori = preprocess_image('data/test/images/{}'.format(ARGS.source_image), 1)

outputImg, _ = my_network(src_img, outputSH, 0)


outputImg = outputImg[0].cpu().data.numpy()
outputImg = outputImg.transpose((1,2,0))
outputImg = np.squeeze(outputImg)
outputImg = (outputImg*255.0).astype(np.uint8)
Lab[:,:,0] = outputImg
resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
resultLab = cv2.resize(resultLab, (col, row))
img_name, e = os.path.splitext(ARGS.source_image)
if (ARGS.face_detect == 'both'):
    img_name += "_faceDetectBoth"
if (ARGS.face_detect == 'light'):
    img_name += "_faceDetectLight"

# calculate and apply mask
crop_paste = ARGS.crop_paste
print(crop_paste)
if crop_paste:
    for i in range(10):
        mask = generate_mask(resultLab, img_name, i)
        resultTemp = apply_mask(resultLab[:,:,:], img_ori, mask)

        cv2.imwrite(os.path.join(saveFolder,
                '{}_relit_{}.jpg'.format(img_name, i)), resultTemp)
    
cv2.imwrite(os.path.join(saveFolder,
        '{}_relit.jpg'.format(img_name)), resultLab)
#----------------------------------------------