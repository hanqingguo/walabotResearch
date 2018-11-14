import cv2
import os


data = {"activity":"walk",
        "video_name":"2.avi",
        "cut_from": 5,
        "cut_length": 10
        }
#
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cur_dir = os.path.dirname(os.path.realpath(__file__))
#/home/hanqing/walabot_Research/walabotResearch/python

root_path = os.path.dirname(cur_dir)
#/home/hanqing/walabot_Research/walabotResearch

dataset_name = "test_dataset"
dataset = os.path.join(root_path, dataset_name)

out_dataset_name = "cut_dataset"
out_dataset = os.path.join(root_path, out_dataset_name)

video_dataset = os.path.join(dataset, data["activity"])
video_path = os.path.join(video_dataset, data["video_name"])
out_name = data["video_name"][:-4]+"_cut.avi"
out_video_dataset = os.path.join(out_dataset, data["activity"])


video_path = os.path.join(video_dataset, data['video_name'])
out_path = os.path.join(out_video_dataset, out_name)
print(video_path, out_path)

# out = cv2.VideoWriter("/home/hanqing/walabot_Research/walabotResearch/test_dataset/walk/23_cut.avi", fourcc, 1.0, (703, 576))
#
# cap = cv2.VideoCapture("/home/hanqing/walabot_Research/walabotResearch/test_dataset/walk/2.avi")

out = cv2.VideoWriter(out_path, fourcc, 1.0, (703, 576))

cap = cv2.VideoCapture(video_path)

n = 0
while cap.isOpened() and n < data["cut_from"] + data["cut_length"] + 1:
    _, frame = cap.read()
    if (n>data["cut_from"] and n<data["cut_from"]+data["cut_length"]):
        out.write(frame)
        print("write {}".format(n))
    n = n + 1




