import numpy as np
import cv2

cap = cv2.VideoCapture('../../training/stand-to-sit/1.avi')
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
        frame = cv2.flip(frame,0)
        # write the flipped frame
        print(frame.shape)
        #out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()