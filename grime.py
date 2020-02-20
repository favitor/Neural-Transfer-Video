import cv2
import sys
import numpy as np
import argparse

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image',
                type=str,
                help='Path to the image.')


#img = cv2.imread("trust4.jpg")
img = cv2.imread(FLAGS.image)


height = img.shape[0]
width = img.shape[1]
width_cutoff = width // 2
face1 = img[:, :width_cutoff]
face2 = img[:, width_cutoff:]


stylize = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
stylizes1 = cv2.stylization(s1, sigma_s=60, sigma_r=0.07)
#Can limit the amount of shade in the output by varying the shade_factor in the function cv2.pencilSketch.
#sketch_gray, sketch_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.06)
sketch_gray1, sketch_color1 = cv2.pencilSketch(face2, sigma_s=60, sigma_r=0.07, shade_factor=0.06)
#night = cv2.imread("night.png")
#night = cv2.resize(night, (428, 1200))

vis = np.concatenate((face1, sketch_color1), axis=1)

#cv2.imwrite("transfer.jpg", vis)
cv2.imwrite("transfer1.jpg", sketch_color1)
cv2.waitKey(0)
cv2.destroyAllWindows()
effect_type = input(int('''Chose a effect:
    [1] Sketch 
    [2] Stylize'''))

if effect_type == 1:
    sketch_color = cv2.pencilSketch(face2, sigma_s=60, sigma_r=0.07, shade_factor=0.06)

elif effect_type == 2:
    stylizes = cv2.stylization(face1, sigma_s=60, sigma_r=0.07)

else
    print("Error, please try again.")  

cv2.imshow("Final Image", vis)
img_save = input(str("Do you want save the image? [Y/N]")).lower()
if img_save == 'y' or 'Y':
    cv2.imwrite("transfer1.jpg", sketch_color1)
    print("Image saved.")
cv2.waitKey(0)
cv2.destroyAllWindows()