import cv2
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
#/home/hanqing/walabot_Research/walabotResearch/python/testThing

root_path = os.path.dirname(os.path.dirname(cur_dir))
#/home/hanqing/walabot_Research/walabotResearch

dataset = os.path.join(root_path, 'test_dataset')

activity = 'stand-to-sit'

video_dataset = os.path.join(dataset, activity)

#/home/hanqing/walabot_Research/walabotResearch/test_dataset/stand-to-sit

videos = os.listdir(video_dataset)

num_pict = 0

for video in videos:
    if num_pict>122:
        break
    else:

        video_path = os.path.join(video_dataset, video)
        cap = cv2.VideoCapture(video_path)
        n = 0
        while cap.isOpened() and n<15:
            _, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(100)
            save_path = os.path.join(os.path.dirname(os.path.dirname(cur_dir)),'training_backup/Classfier/Test/walk')
            #/home/hanqing/walabot_Research/walabotResearch/training_backup/Classfier/walk
            save_video = os.path.join(save_path, video[:-4]+'-'+str(n)+".jpg")
            print(save_video)
            cv2.imwrite(save_video, frame)
            n +=1
            num_pict += 1
            print(num_pict)


