import cv2
vidcap = cv2.VideoCapture('4Years.mp4')
success,image = vidcap.read()
count = 0
success = True
while success is True:
  cv2.imwrite("frame%d.jpg" % count, image)
  success, image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
print("End")