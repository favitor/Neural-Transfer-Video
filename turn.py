import cv2 as cv
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('./styleimages/*.jpg'):
    img = cv.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv.VideoWriter('project.avi',cv.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()