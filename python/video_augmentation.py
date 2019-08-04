import cv2
import os
import os.path as osp

curdir = os.path.dirname(os.path.realpath('.'))
dataset_path = osp.join(curdir, 'cut_dataset')
print(dataset_path)
files = os.listdir(dataset_path)
print(files)

activity = osp.join(dataset_path, files[1])
print(activity)

def reverse(video, out_path):


    video = osp.join(activity, video)

    out_path = video[:-4]+'_'+out_path +'.avi'
    print(out_path)


    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_path, fourcc, 1.0, (703, 576))
    cap = cv2.VideoCapture(video)
    frames = []

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        frames.append(frame)
    for rev_frame in reversed(frames):
        out.write(rev_frame)

for video in os.listdir(activity):
    reverse(video, "REV")



