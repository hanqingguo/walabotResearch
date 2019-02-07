from __future__ import print_function  # WalabotAPI works on both Python 2 an 3.
from sys import platform
from os import system
from imp import load_source
from os.path import join
import numpy as np
import time
import matplotlib.pyplot as plt
import io
import PIL
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
import cv2
import os
from python.loadModel import *
import torch

if platform == 'win32':
    modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
                      'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
    modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()

current_dir = os.path.dirname(os.path.realpath(__file__))
training_dir = os.path.join(os.path.dirname(current_dir), 'training')

"""

current_dir  = /home/hanqing/walabot_Research/walabotResearch/python
training_dir = /home/hanqing/walabot_Research/walabotResearch/training

"""

activities = ['walk', 'sit-to-stand', 'stand-to-sit', 'fall_down', 'jump']

idx = 0
video_name = 'nan'

training_path = os.path.join(training_dir, activities[idx])
if activities[idx] not in os.listdir(training_dir):
    os.mkdir(training_path, mode=0o777)

video_path = os.path.join(training_path, '{}.avi'.format(video_name))

def PrintSensorTargets(targets):
    system('cls' if platform == 'win32' else 'clear')
    if targets:
        for i, target in enumerate(targets):
            print('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                i + 1, target.xPosCm, target.yPosCm, target.zPosCm,
                target.amplitude))
    else:
        print('No Target Detected')

def normlize(img):
    img = (img-np.min(img)) / (np.max(img) - np.min(img))
    return img


def plot_3d(image, minInCm, resInCm, minPhiInDegrees, resPhiInDegrees, minThetaIndegrees, resThetaIndegrees, threshold=0):

    # stack sliced image to upright
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, level=threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Make change axis from 3d volume indexes , to more make sense data
    # (0,100) => (1,200)  R(cm)
    # (0,61)  => (-90,90) Phi(degree)
    # (0,9)   => (-20,20) Theta(degree)

    verts[:,0] = minInCm + verts[:,0] * resInCm
    verts[:,1] = verts[:,1] * resPhiInDegrees + minPhiInDegrees
    verts[:,2] = verts[:,2] * resThetaIndegrees + minThetaIndegrees

    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    #print(mesh)

    face_color = [0.8, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(minInCm, p.shape[0]*resInCm)
    ax.set_ylim(minPhiInDegrees, p.shape[1]*resPhiInDegrees + minPhiInDegrees)
    ax.set_zlim(minThetaIndegrees, p.shape[2]*resThetaIndegrees + minThetaIndegrees)

    #print(current_xticks)
    #ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_xlabel("Direct Distance (cm)")
    ax.set_ylabel("Phi (degree)")
    ax.set_zlabel("Theta (degree)")



    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png", bbox_inches='tight', pad_inches=1)
    buffer_.seek(0)

    image = PIL.Image.open(buffer_)

    ar = np.asarray(image)
    plt.close()


    return ar


def SensorApp():
    # wlbt.SetArenaR - input parameters
    minInCm, maxInCm, resInCm = 1, 200, 2
    # wlbt.SetArenaTheta - input parameters
    minThetaIndegrees, maxThetaIndegrees, resThetaIndegrees = -30, 30, 3
    # wlbt.SetArenaPhi - input parameters
    minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -60, 60, 3
    # Set MTI mode
    mtiMode = False
    # Configure Walabot database install location (for windows)
    wlbt.SetSettingsFolder()
    # 1) Connect : Establish communication with walabot.
    wlbt.ConnectAny()
    # 2) Configure: Set scan profile and arena
    # Set Profile - to Sensor.
    wlbt.SetProfile(wlbt.PROF_SENSOR)
    # Setup arena - specify it by Cartesian coordinates.
    wlbt.SetArenaR(minInCm, maxInCm, resInCm)
    # Sets polar range and resolution of arena (parameters in degrees).
    wlbt.SetArenaTheta(minThetaIndegrees, maxThetaIndegrees, resThetaIndegrees)
    # Sets azimuth range and resolution of arena.(parameters in degrees).
    wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
    # Moving Target Identification: standard dynamic-imaging filter
    filterType = wlbt.FILTER_TYPE_MTI if mtiMode else wlbt.FILTER_TYPE_NONE
    wlbt.SetDynamicImageFilter(filterType)
    # 3) Start: Start the system in preparation for scanning.
    wlbt.Start()
    if not mtiMode:  # if MTI mode is not set - start calibrartion
        # calibrates scanning to ignore or reduce the signals
        wlbt.StartCalibration()
        while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
            wlbt.Trigger()


    start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, 5.0, (703, 576))
    frame_count = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ori_model = models.resnet18(pretrained=True)
    model = CNN(ori_model)
    model.load_state_dict(torch.load("../python/classfier"))
    model = model.to(device)



    while frame_count < 1000:
        appStatus, calibrationProcess = wlbt.GetStatus()
        # 5) Trigger: Scan(sense) according to profile and record signals
        # to be available for processing and retrieval.
        wlbt.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        # rasterImage, _, _, sliceDepth, power = wlbt.GetRawImageSlice()
        # PrintSensorTargets(targets)
        rasterImage, sizeX, sizeY, sizeZ, power = wlbt.GetRawImage()
        """
            sizeX = (maxTheta - minTheta) / resTheta
            sizeY = (maxPhiInDegrees - minPhiInDegrees) / resPhiInDegrees
            sizeZ = (maxInCm - minInCm) / resInCm
        """
        # PrintSensorTargets(targets)
        img = np.array(rasterImage)

        img = normlize(img)

        frame = plot_3d(img, minInCm, resInCm, minPhiInDegrees, resPhiInDegrees, minThetaIndegrees, resThetaIndegrees,threshold=0.8)

        # change from 4 channel image to 3 channel image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # filter bad frame
        result = visualize_stream(model, frame)


        if (result):
            cv2.imshow("frame", frame)
            cv2.waitKey(5)

        frame_count += 1


    # 7) Stop and Disconnect.
    out.release()
    wlbt.Stop()
    wlbt.Disconnect()
    print('Terminate successfully')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SensorApp()
