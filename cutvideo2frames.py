import numpy as np
import cv2
from PIL import Image
import os


def deal_video():
    cap = cv2.VideoCapture('data/cinesmall.avi')
    a = 0
    frams = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        a+=1


        # pic.show()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pic = Image.fromarray(frame,"RGB")
            # pic = pic.resize((64,64))
            pic.save('data/cine/%d.png'%a)
            # cv2.imshow('frame',frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    deal_video()