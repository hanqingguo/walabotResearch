import cv2
import os


data = {"activity":"fall_down",
        "video_name":"fall_down_3.avi",
        "cut_from": 60,
        "cut_length": 6,
        "out_idx": "0"
        }
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cur_dir = os.path.dirname(os.path.realpath(__file__))
#/home/hanqing/walabot_Research/walabotResearch/python

root_path = os.path.dirname(cur_dir)
#/home/hanqing/walabot_Research/walabotResearch

dataset_name = "training"
dataset = os.path.join(root_path, dataset_name)

out_dataset_name = "cut_dataset"
out_dataset = os.path.join(root_path, out_dataset_name)

video_dataset = os.path.join(dataset, data["activity"])
video_path = os.path.join(video_dataset, data["video_name"])
# comment below line only cut stand_up category
# out_name = data["video_name"][:-4]+"_cut5.avi"
out_name = "stand_up_"+"_cut5.avi"
print(out_name)

out_video_dataset = os.path.join(out_dataset, data["activity"])


video_path = os.path.join(video_dataset, data['video_name'])
out_path = os.path.join(out_video_dataset, out_name)
print(video_path)

print(out_path)


# video_path = "/home/hanqing/walabot_Research/walabotResearch/training/walk/walk_0.avi"
# out_path = "/home/hanqing/walabot_Research/walabotResearch/cut_dataset/walk/walk_0_cut.avi"


out = cv2.VideoWriter(out_path, fourcc, 1.0, (703, 576))

cap = cv2.VideoCapture(video_path)

n = 0
while cap.isOpened() and n < data["cut_from"] + data["cut_length"] + 1:
    _, frame = cap.read()
    if (n>data["cut_from"] and n<data["cut_from"]+data["cut_length"] + 1):
        out.write(frame)
        print("write {}".format(n))
    n = n + 1




