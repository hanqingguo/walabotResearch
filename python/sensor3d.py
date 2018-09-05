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

if platform == 'win32':
    modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
                      'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
    modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()


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


def plot_3d(image, threshold=0):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera

    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, level=threshold)

    # print(verts, faces)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png", bbox_inches='tight', pad_inches=0)
    buffer_.seek(0)

    image = PIL.Image.open(buffer_)

    ar = np.asarray(image)

    return ar


def SensorApp():
    # wlbt.SetArenaR - input parameters
    minInCm, maxInCm, resInCm = 1, 200, 2
    # wlbt.SetArenaTheta - input parameters
    minThetaIndegrees, maxThetaIndegrees, resThetaIndegrees = -20, 20, 5
    # wlbt.SetArenaPhi - input parameters
    minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -90, 90, 3
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
    sliceDepths = []
    imgs = []

    start_time = time.time()
    while time.time() - start_time < 60:
        appStatus, calibrationProcess = wlbt.GetStatus()
        # 5) Trigger: Scan(sense) according to profile and record signals
        # to be available for processing and retrieval.
        wlbt.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        targets = wlbt.GetSensorTargets()
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

        # print(img.shape)
        # print("sizeX is {}".format(sizeX))
        # print("sizeY is {}".format(sizeY))
        # print("sizeZ is {}".format(sizeZ))

        img = normlize(img)

        frame = plot_3d(img, 0.8)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # camp = plt.get_cmap("hot")
        # norm_img = (img - np.min(img)) / (np.max(img) - np.min(img)) # normalize all dimensions
        # for i in range(norm_img.shape[0]):
        #     rgba_img = camp(norm_img[i, :, :])
        #     rgb_img = np.delete(rgba_img, 3, -1)
        #     print("shape of single rgb_img is: {}".format(rgb_img.shape))
        #
        #     plot_3d(rgb_img, 0.5)
        #
        #     plt.show()


    # 7) Stop and Disconnect.

    wlbt.Stop()
    wlbt.Disconnect()
    print('Terminate successfully')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SensorApp()
