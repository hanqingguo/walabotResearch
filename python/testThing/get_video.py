import numpy as np
import cv2

cap = cv2.VideoCapture('../../training_backup/training/stand-to-sit/5.avi')
# print(cap.get(3))
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (320,240))

n=0
while(cap.isOpened()):
    ret, frame = cap.read()
    n = n + 1
    print(n)
    if ret==True:

        cv2.imshow('frame',frame)
        cv2.waitKey(1000)
    else:
        break

# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()