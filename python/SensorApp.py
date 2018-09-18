from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
from sys import platform
from os import system
from imp import load_source
from os.path import join
import numpy as np
import time
import matplotlib.pyplot as plt


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

def SensorApp():
    # wlbt.SetArenaR - input parameters
    minInCm, maxInCm, resInCm = 1, 200, 2
    # wlbt.SetArenaTheta - input parameters
    minThetaIndegrees, maxThetaIndegrees, resThetaIndegrees = -15, 15, 5
    # wlbt.SetArenaPhi - input parameters
    minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -90, 90, 1
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
    if not mtiMode: # if MTI mode is not set - start calibrartion
        # calibrates scanning to ignore or reduce the signals
        wlbt.StartCalibration()
        while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
            wlbt.Trigger()
    sliceDepths = []
    imgs = []

    start_time = time.time()
    while time.time() - start_time < 20:
        appStatus, calibrationProcess = wlbt.GetStatus()
        # 5) Trigger: Scan(sense) according to profile and record signals
        # to be available for processing and retrieval.
        wlbt.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        targets = wlbt.GetSensorTargets()
        #rasterImage, _, _, sliceDepth, power = wlbt.GetRawImageSlice()
        # PrintSensorTargets(targets)
        rasterImage, sizeX, sizeY, sliceDepth, power = wlbt.GetRawImageSlice()
        """
            sizeX = (maxInCm - minInCm) / resInCm
            sizeY = (maxPhiInDegrees - minPhiInDegrees) / resPhiInDegrees
        """
        #PrintSensorTargets(targets)
        img = np.array(rasterImage)
        plt.imshow(img, cmap=plt.cm.hot, interpolation='nearest', extent=[-90,90,200,0])
        # Thanks https://stackoverflow.com/questions/18696122/change-values-on-matplotlib-imshow-graph-axis
        # answer how to change plt.imshow axis
        plt.xlabel("Phi(degree)")


        #print(x_min, x_max)

        plt.ylabel("R(cm)")
        #plt.yscale(2)

        plt.show()
        imgs.append(img)
        print(img.shape)
        print("sizeX is {}".format(sizeX))
        print("sizeY is {}".format(sizeY))
        
        #sliceDepths.append(sliceDepth)
        print("sliceDepth is {}".format(sliceDepth))
        #print("power is {}".format(power))
    # 7) Stop and Disconnect.

    # plt.imshow(imgs[5])
    # plt.colorbar()
    # plt.show()
    wlbt.Stop()
    wlbt.Disconnect()
    print('Terminate successfully')

if __name__ == '__main__':
    SensorApp()
