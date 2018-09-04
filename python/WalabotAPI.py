from __future__ import unicode_literals
from sys import platform, maxsize
from ctypes import *
from os.path import exists, join
from collections import namedtuple

def _GetDefaultPaths(): 
    if platform == 'win32':
        libPath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
            'bin', 'Win32', 'WalabotAPIWindows.dll')
        if maxsize > 2**32: libPath = libPath.replace('Win32', 'x64')
        settingsFolderPath = join('C:/', 'ProgramData', 'Walabot', 'WalabotSDK')
    elif platform.startswith('linux'):
        libPath = join('/usr', 'lib', 'libWalabotAPI.so')     
        settingsFolderPath = join('/var', 'lib', 'walabot')  
    else:
        return None, None
    return libPath, settingsFolderPath
_defaultLibPath, _defaultSettingsFolderPath = _GetDefaultPaths()

def Init(libPath = _defaultLibPath):
    """Must be called before using WalabotAPI functions.
        Args:
            4:        Full path to Walabot shared library. If not set, will use default path.
    """
    if not exists(libPath):
        raise ValueError('Could not load Walabot library at:', libPath)
    global _wlbt
    _wlbt = CDLL(libPath)

def IsInitialized():
    return _wlbt is not None

# Walabot constant values
PROF_SHORT_RANGE_IMAGING = 0x00010000
PROF_SENSOR = 0x00020000
PROF_SENSOR_NARROW = 0x00020000
STATUS_DISCONNECTED = 0
STATUS_CONNECTED = 1
STATUS_IDLE = 2
STATUS_SCANNING = 3
STATUS_CALIBRATING = 4
TARGET_TYPE_UNKNOWN = 0
TARGET_TYPE_PIPE = 1
FILTER_TYPE_NONE = 0
FILTER_TYPE_DERIVATIVE = 1
FILTER_TYPE_MTI = 2

WALABOT_SUCCESS = 0
WALABOT_ERR_INVALID_PORT_DEFINITION = 1
WALABOT_ERR_TOO_MANY_SAMPLES = 2
WALABOT_ERR_RECORDING_THREAD_ALREADY_RUNNING = 3
WALABOT_ERR_RECORDING_WITHOUT_APPLY_SETTINGS = 4
WALABOT_ERR_READ_WITHOUT_APPLY_SETTINGS = 5
WALABOT_ERR_INVALID_PAGE_INDEX = 6
WALABOT_ERR_INVALID_TXRX_PAIR = 7
WALABOT_ERR_AMBIGUOUS_TXRX_PAIR = 8
WALABOT_ERR_PORTGROUP_INDEX_OUT_OF_BOUNDS = 9
WALABOT_ERR_RX_PORT_OUT_OF_BOUNDS = 10
WALABOT_ERR_NO_DATA = 11
WALABOT_ERR_USB_READ_FAILURE = 12
WALABOT_PARAMETER_NOT_FOUND = 13
WALABOT_OUT_OF_RANGE = 14
WALABOT_READONLY_PARAMETER = 15
WALABOT_ERR_ARENA_SIZE_TOO_BIG = 16
WALABOT_DD_ERR_USB_DEVICE_WENT_DOWN = 17
WALABOT_ERR_INVALID_PARAMETER = 18
WALABOT_INSTRUMENT_NOT_FOUND = 19
WALABOT_BUFFER_TOO_SMALL = 20
WALABOT_INVALID_ARRAY_SIZE = 21
WALABOT_INVALID_PROFILE = 22
WALABOT_ERR_NO_IMAGE = 23
WALABOT_ERR_GENERAL = 24
WALABOT_ERR_PROTOCOL = 0x00010000
WALABOT_ERR_DEVICE_STORAGE = 0x00020000
WALABOT_ERR_DB = 0x00030000
WALABOT_ERR_USB = 0x00040000
WALABOT_ERR_SEQ = 0x00050000
WALABOT_ERR_SCHEDULER = 0x00070000
WALABOT_ERR_HW_PLAN = 0x00080000
WALABOT_ERR_MULTI_CHIP = 0x00090000
WALABOT_ERR_REC_PLAN = 0x000A0000
WALABOT_ERR_NULL_OBJECT = 0x000B0000
WALABOT_ERR_IN_COMPILER = 0x000C0000
WALABOT_ERR_IN_INITIALIZATION = 0x000D0000
WALABOT_ERR_SETTING_FOLDER = 0x000E0000
WALABOT_ERR_LOG = 0x000F0000
WALABOT_GENERAL_ERROR = 0x000F0000 + 1

AppStatus = [STATUS_CALIBRATING, STATUS_CONNECTED, STATUS_DISCONNECTED,
    STATUS_IDLE, STATUS_SCANNING]
AppProfile = [PROF_SENSOR, PROF_SENSOR_NARROW, PROF_SHORT_RANGE_IMAGING]
FilterType = [FILTER_TYPE_DERIVATIVE, FILTER_TYPE_MTI, FILTER_TYPE_NONE]

WalabotResult = [v for k, v in locals().items() if k[:7] == 'WALABOT']
assert(len(WalabotResult)==len(set(WalabotResult))) # check WalabotResult values are unique
_WalabotResultDict = {v:k for k, v in locals().items() if k[:7] == 'WALABOT'}

AntennaPair = namedtuple("AntennaPair", "txAntenna rxAntenna")
ImagingTarget = namedtuple("ImagingTarget",
    "type angleDeg xPosCm yPosCm zPosCm widthCm amplitude")
SensorTarget = namedtuple('SensorTarget', 'xPosCm yPosCm zPosCm amplitude')

class WalabotError(Exception):
    """ Walabot's specific Exception object.
        Args:
            message:        Short explanation about the occured exception.
            code:           Code number of the exception.
            extended:       Error information for Vayyar support.
    """
    def __init__(self, message, code, extended):
        super(Exception, self).__init__(message)
        self.code = code
        self.extended = extended

def _RaiseIfErr(res):
    """ Raises customized WalabotError in case encounter one.
    """
    if res != 0:
        message = _WalabotResultDict[res]
        extended = _wlbt.Walabot_GetExtendedError()
        raise WalabotError(message, res, extended)

def _SetArenaHelper(func, minValue, maxValue, resValue):
    """ Converts the arguments to 'ctypes' types, then calls the desired func.
    """
    func.argtypes = [c_double, c_double, c_double]
    _RaiseIfErr(func(minValue, maxValue, resValue))

def _GetArenaHelper(func):
    """ Converts the arguments to 'ctypes' types, calls the desired func,
        returns results as 'Pythonic variables'.
    """
    minValue, maxValue, resValue = c_double(), c_double(), c_double()
    _RaiseIfErr(func(byref(minValue), byref(maxValue), byref(resValue)))
    return minValue.value, maxValue.value, resValue.value

def GetExtendedError():
    """ Obtains additional error information, for Vayyar support.
    """
    return _wlbt.Walabot_GetExtendedError()

def GetErrorString():
    """Obtains the detailed string of the last error."""
    f = _wlbt.Walabot_GetErrorString
    f.restype = c_char_p
    return f().decode()

def SetSettingsFolder(path = _defaultSettingsFolderPath):
    """ Obtains Sets location of Walabot internal database, if moved from
        default.
        Args:
            path           (Optional) Database location. Uses default location
                            if no path is given.
    """
    _RaiseIfErr(_wlbt.Walabot_SetSettingsFolder(path.encode('ascii')))

def GetInstrumentsList():
    """Obtains a string listing connected Walabots.

    For use before WalabotAPI.Connect(). A Walabot ID obtained here is used to identify the device to connect to.
    """
    default_buf_len = (1<<10)
    buf = create_string_buffer(default_buf_len)
    _RaiseIfErr(_wlbt.Walabot_GetInstrumentsList(default_buf_len, buf))
    return buf.value

def Connect(uid):
    """Establishes communication with a specified Walabot device.
    
    Connection is required before WalabotAPI.Start().
    If only a single Walabot device is present, it is simpler to use WalabotAPI.ConnectAny(), so the uid is not required.
    
        Parameters:
            uid             Walabot device ID string, obtained from
                            GetInstrumentsList()
    """
    _RaiseIfErr(_wlbt.Walabot_Connect(str.encode(uid)))

def ConnectAny():
    """ Establishes communication with Walabot.
        Connection is required before Start(). If multiple Walabots are
        present a single available Walabot is selected. To specify one, use
        Connect().
    """
    _RaiseIfErr(_wlbt.Walabot_ConnectAny())

def Disconnect():
    """ Stops communication with Walabot.
    """
    _RaiseIfErr(_wlbt.Walabot_Disconnect())

def Start():
    """ Starts Walabot in preparation for scanning.
        Requires previous Connect (ConnectAny() or Connect()) and SetProfile().
        Required before Trigger() and GET actions.
    """
    _RaiseIfErr(_wlbt.Walabot_Start())

def Stop():
    """ Stops Walabot when finished scanning.
    """
    _RaiseIfErr(_wlbt.Walabot_Stop())

def Trigger():
    """ Initiates a scan and records signals.
        Initiates a scan according to profile and records signals to be
        available for processing and retrieval. Should be performed before
        every GET action.
    """
    _RaiseIfErr(_wlbt.Walabot_Trigger())

class _Ctypes_AntennaPairs(Structure):
    _fields_ = [('txAntenna', c_int), ('rxAntenna', c_int)]

def GetAntennaPairs():
    """ Obtains a list of Walabot antenna pairs.
        For use before GetSignal(). To identify the antennas on your Walabot
        device, see the specifications for your model.
        Returns:
            antennaPairs:   List of antenna pairs (of AntennaPair namedtuple)
    """
    antPairs, numPairs = pointer(_Ctypes_AntennaPairs()), c_int()
    _RaiseIfErr(_wlbt.Walabot_GetAntennaPairs(byref(antPairs), byref(numPairs)))
    return [AntennaPair(antPairs[i].txAntenna, antPairs[i].rxAntenna)
        for i in range(numPairs.value)]

def GetSignal(antennaPair):
    """ Obtains raw image data from specified antennas.
        Args:
            antennaPair:    Transmitting antenna ID and receiving antenna ID
                            as obtained from GetAntennaPairs()
        Returns:
            signal:         List of amplitude values representing received
                            signal amplitude in time domain
            timeAxis:       List of time axis ticks values
    """
    signal, timeAxis, numSmp = pointer(c_double()), pointer(c_double()), c_int()
    txAntenna, rxAntenna = c_int(antennaPair.txAntenna), c_int(antennaPair.rxAntenna)
    _RaiseIfErr(_wlbt.Walabot_GetSignal(txAntenna, rxAntenna,
        byref(signal), byref(timeAxis), byref(numSmp)))
    signal_list = [signal[i] for i in range(numSmp.value)]
    timeAxis_list = [timeAxis[i] for i in range(numSmp.value)]
    return signal_list, timeAxis_list

def SetProfile(appProfile):
    """ Sets scan profile.
        Args:
            appProfile:     The scan profile to use
    """
    _RaiseIfErr(_wlbt.Walabot_SetProfile(appProfile))

def GetStatus():
    """ Obtains Walabot status.
        Returns:
            appStatus       Walabot's status
            param           Percentage of calibration completed, if status
                            is STATUS_CALIBRATING
    """
    appStatus, param = c_int(), c_double()
    _RaiseIfErr(_wlbt.Walabot_GetStatus(byref(appStatus), byref(param)))
    return appStatus.value, param.value

def StartCalibration():
    """ Initiates calibration.
        Ignores or reduces the signals of fixed reflectors such as walls
        according to environment. Must be performed initially (to avoid delays
        preferably before Start()), upon any profile change, and is recommended
        upon possible changes to environment. Calibration is done via recording
        and processing. So after calling StartCalibration, a continues trigger
        & GetImage/GetTargets function calls is required. To check on
        calibration progress, use GetStatus().
    """
    _RaiseIfErr(_wlbt.Walabot_StartCalibration())

def CancelCalibration():
    """ Stops calibration.
        To check on calibration progress, use GetStatus().
    """
    _RaiseIfErr(_wlbt.Walabot_CancelCalibration())

def SetThreshold(threshold):
    """ Changes the sensitivity threshold.
        For raw images (3-D and Slice), Walabot removes very weak signals,
        below this threshold. If the threshold is not set, a default value is
        used. To check the current value, use GetThreshold().
        Args:
            threshold       The threshold to set
    """
    _RaiseIfErr(_wlbt.Walabot_SetThreshold(c_double(threshold)))

def GetThreshold():
    """ Obtains the current sensitivity threshold.
        To set the threshold, use SetThreshold().
        Returns:
            threshold:      The current threshold value
    """
    threshold = c_double()
    _RaiseIfErr(_wlbt.Walabot_GetThreshold(byref(threshold)))
    return threshold.value

def SetArenaX(minInCm, maxInCm, resInCm):
    """ Sets X-axis range and resolution of arena.
        To check the current value, use GetArenaX(). Note: Cartesian (X-Y-Z)
        coordinates should be used only to get image data from a triggered scan
        that used the short-range profile. Otherwise use the SetArena functions
        for spherical coordinates.
        Args:
            minInCm         Beginning of range on axis (cm)
            maxInCm         End of range on axis (cm)
            resInCm         Distance between pixels along axis (cm)
    """
    _SetArenaHelper(_wlbt.Walabot_SetArenaX, minInCm, maxInCm, resInCm)

def SetArenaY(minInCm, maxInCm, resInCm):
    """ Sets Y-axis range and resolution of arena.
        To check the current value, use GetArenaY(). Note: Cartesian (X-Y-Z)
        coordinates should be used only to get image data from a triggered scan
        that used the short-range profile. Otherwise use the SetArena functions
        for spherical coordinates.
        Args:
            minInCm         Beginning of range on axis (cm)
            maxInCm         End of range on axis (cm)
            resInCm         Distance between pixels along axis (cm)
    """
    _SetArenaHelper(_wlbt.Walabot_SetArenaY, minInCm, maxInCm, resInCm)

def SetArenaZ(startInCm, endInCm, resInCm):
    """ Sets Z-axis range and resolution of arena.
        To check the current value, use GetArenaZ(). Note: Cartesian (X-Y-Z)
        coordinates should be used only to get image data from a triggered scan
        that used the short-range profile. Otherwise use the SetArena functions
        for spherical coordinates.
        Args:
            startInCm       Beginning of range on axis (cm)
            endInCm         End of range on axis (cm)
            resInCm         Distance between pixels along axis (cm)
    """
    _SetArenaHelper(_wlbt.Walabot_SetArenaZ, startInCm, endInCm, resInCm)

def SetArenaR(startInCm, endInCm, resInCm):
    """ Sets radial range and resolution of arena.
        To check the current value, use GetArenaR(). Note: Cartesian (X-Y-Z)
        coordinates should be used only to get image data from a triggered scan
        that used the short-range profile. Otherwise use the SetArena functions
        for spherical coordinates.
        Args:
            startInCm       Beginning of range on axis (cm)
            endInCm         End of range on axis (cm)
            resInCm         Distance between pixels along axis (cm)
    """
    _SetArenaHelper(_wlbt.Walabot_SetArenaR, startInCm, endInCm, resInCm)

def SetArenaTheta(minInDegrees, maxInDegrees, resInDegrees):
    """ Sets polar range and resolution of arena.
        To check the current value, use GetArenaTheta(). Spherical coordinates
        should be used only to get image data from a triggered scan that used
        one of the Sensor profiles. Otherwise use the SetArena functions for
        cartesian coordinates.
        Args:
            minInDegrees    Beginning of polar angular range (degrees)
            maxInDegrees    End of polar angular range (degrees)
            resInDegrees    Angle between pixels across polar angle (degrees)
    """
    _SetArenaHelper(_wlbt.Walabot_SetArenaTheta, minInDegrees,
        maxInDegrees, resInDegrees)

def SetArenaPhi(minInDegrees, maxInDegrees, resInDegrees):
    """ Sets polar range and resolution of arena.
        To check the current value, use GetArenaTheta(). Spherical coordinates
        should be used only to get image data from a triggered scan that used
        one of the Sensor profiles. Otherwise use the SetArena functions for
        cartesian coordinates.
        Args:
            minInDegrees    Beginning of polar angular range (degrees)
            maxInDegrees    End of polar angular range (degrees)
            resInDegrees    Angle between pixels across polar angle (degrees)
    """
    _SetArenaHelper(_wlbt.Walabot_SetArenaPhi, minInDegrees,
        maxInDegrees, resInDegrees)

def GetArenaX():
    """ Obtains current X-axis range and resolution of arena.
        Can be changed with SetArenaX(). Cartesian coordinates are relevant
        only to get image data from a triggered scan that used the short-range
        profile. Otherwise the SetArena functions for spherical coordinates
        apply.
        Returns:
            minInCm:        Beginning of range on axis (cm)
            maxInCm:        End of range on axis (cm)
            resInCm:        Distance between pixels along axis (cm)
    """
    return _GetArenaHelper(_wlbt.Walabot_GetArenaX)

def GetArenaY():
    """ Obtains current Y-axis range and resolution of arena.
        Can be changed with SetArenaY(). Cartesian coordinates are relevant
        only to get image data from a triggered scan that used the short-range
        profile. Otherwise the SetArena functions for spherical coordinates
        apply.
        Returns:
            minInCm:        Beginning of range on axis (cm)
            maxInCm:        End of range on axis (cm)
            resInCm:        Distance between pixels along axis (cm)
    """
    return _GetArenaHelper(_wlbt.Walabot_GetArenaY)

def GetArenaZ():
    """ Obtains current Z-axis range and resolution of arena.
        Can be changed with SetArenaZ(). Cartesian coordinates are relevant
        only to get image data from a triggered scan that used the short-range
        profile. Otherwise the SetArena functions for spherical coordinates
        apply.
        Returns:
            startInCm:      Beginning of range on axis (cm)
            endInCm:        End of range on axis (cm)
            resInCm:        Distance between pixels along axis (cm)
    """

    return _GetArenaHelper(_wlbt.Walabot_GetArenaZ)

def GetArenaR():
    """ Obtains radial range and resolution of arena.
        Can be changed with SetArenaR(). Spherical coordinates are relevant
        only to get image data from a triggered scan that used one of the
        Sensor profiles. Otherwise the SetArena functions for cartesian
        coordinates apply.
        Returns:
            startInCm:      Beginning of radial distance range (cm)
            endInCm:        End of radial distance range (cm)
            resInCm:        Image resolution along radius (cm)
    """
    return _GetArenaHelper(_wlbt.Walabot_GetArenaR)

def GetArenaTheta():
    """ Obtains polar range and resolution of arena.
        Can be changed with SetArenaTheta(). Spherical coordinates are
        relevant only to get image data from a triggered scan that used one of
        the Sensor profiles. Otherwise the SetArena functions for cartesian
        coordinates apply.
        Returns:
            minInDegrees:   Beginning of polar angular range (degrees)
            maxInDegrees:   End of polar angular range (degrees)
            resInDegrees:   Angle between pixels across polar angle (degrees)
    """
    return _GetArenaHelper(_wlbt.Walabot_GetArenaTheta)

def GetArenaPhi():
    """ Obtains azimuth range and resolution of arena.
        Can be changed with SetArenaPhi(). Spherical coordinates are relevant
        only to get image data from a triggered scan that used one of the
        Sensor profiles. Otherwise the SetArena functions for cartesian
        coordinates apply.
        Returns:
            minInDegrees:   Beginning of azimuth angular range (degrees)
            maxInDegrees:   End of azimuth angular range (degrees)
            resInDegrees:   Angle between pixels across polar angle (degrees)
    """
    return _GetArenaHelper(_wlbt.Walabot_GetArenaPhi)

def GetRawImageSlice():
    """ Provides bidimensional (2-D) image data of the 3D image projected to a plane.

    Image data is a 2-dimensional projection matrix of the 3D Raw image which can be obtained using WalabotAPI.GetRawImage function.

    In this 2D matrix, each element represents the reflection strength at spatial location corresponding to this element indexing in the matrix.
    The projection is done according to the profile used. For sensor profile, the projection is to Phi-R (polar coordinates) plane,
    while for short-range sensor, the projection is to X-Y plane (carthesian coordinates).
    One can always use the original 3D raw data to create other planes of interests.
    The value of the element indicates the reflected power measured in its location.

    The matrix is represented as follows, for 2D index coordinates {i,j}:
        sizeX:  represnts the i dimension length
        sizeY:  represnts the j dimension length
        img3d: represents the walabot 2D scanned image (internal data)
        normalized_abs_val = abs( img3d[i][j] )  ## normalized between 0 to 1
        rasterImage[i][j] = (int)(normalized_abs_val * 255)

    Each index represents the location along its axis according to the arena defined:

        For Carthesian coordinates Arena (using WalabotAPI.SetArenaX(), WalabotAPI.SetArenaY()):
            dx = (xMax-xMin) / (sizeX-1)
            dy = (yMax-yMin) / (sizeY-1)

        Then:
            M[i][j] = strength at position (xMin + i*dx, yMin + j*dy)

        For Polar coordinates Arena ( using WalabotAPI.SetArenaR(), WalabotAPI.SetArenaPhi()):
            dPhi = (phiMax-phiMin) / (sizeX-1)
            dR = (RMax-RMin) / (sizeY-1)
        Then:
            M[i][j] = strength at position (phiMin + i*dPhi, RMin + j*dR)
        Requires previous WalabotAPI.Trigger(). Provides data based on last completed
        triggered image. Output is details of array variable populated by
        provided image data. Provided image data is dependent on current
        configured arena and on current configuration from
        WalabotAPI.SetDynamicImageFilter() and WalabotAPI.SetThreshold().
        Returns:
            rasterImage:    Name of list variable populated by output image data.
            sizeX:          Dimension of list variable populated by output
                            image data.
            sizeY:          Dimension of list variable populated by output
                            image data
            sliceDepth:     Third dimension coordinate of maximum target power.
            power:          Peak measured power in arena (value of strongest
                            pixel).
    """
    carr_rasterImage = POINTER(c_int)()
    sizeX = c_int(0)
    sizeY = c_int(0)
    depth = c_double(0.0)
    power = c_double(0.0)

    res = _wlbt.Walabot_GetRawImageSlice(byref(carr_rasterImage), byref(sizeX), byref(sizeY), byref(depth), byref(power))
    _RaiseIfErr(res)

    X, Y = sizeX.value, sizeY.value
    ## transform 1D C-style array into 2D python list.
    rasterImage = [[carr_rasterImage[(y*X)+x] for y in range(Y)] for x in range(X)]
    return rasterImage, X, Y, depth.value, power.value

def GetRawImage():
    """ Provides tridimensional (3-D) image data.
        Image data is a 3-dimensional matrix in which each element represents
        the reflected power at (x,y,z) spatial location corresponding to this
        element indexing in the matrix. The coordinates are according to the
        profile used. For sensor profile, the coordinates are Theta-Phi-R
        (polar coordinates), while for short-range sensor, the coordinates are
        to X-Y-Z plane (carthesian coordinates). The matrix is transferred
        using a vector which is the concatenated matrix rows. Meaning: assuming
        3D matrix with indexes i, j & k. the vector is represented as followed:
        The matrix is represented as follows, for 3D index coordinates {i,j,k}:
            sizeX:  represnts the i dimension length
            sizeY:  represnts the j dimension length
            sizeZ:  represnts the k dimension length
            img3d:  represtes the walabot 3D scanned image (internal data)
        normalized_abs_val = abs( img3d[i][j][k] ), normalized between 0 to 1
        rasterImage[i][j][k] = (int)(normalized_abs_val * 255)
        Each index represent the location along its axis according to the arena
        defined. For Carthesian coordinates Arena (using WalabotAPI.SetArenaX(),
        WalabotAPI.SetArenaY(), WalabotAPI.SetArenaZ()):
            dx = (xMax-xMin) / (sizeX-1)
            dy = (yMax-yMin) / (sizeY-1)
            dz = (zMax-zMin) / (sizeZ-1)
        Then:
            M[i][j][k] = strength at position (xMin + i*dx, yMin + j*dy, zMin + k*dz)

        For Polar coordinates Arena ( using WalabotAPI.SetArenaR(), WalabotAPI.SetArenaTheta(), WalabotAPI.SetArenaPhi()):
            dTheta = (thetaMax-thetaMin) / (sizeX-1)
            dPhi = (phiMax-phiMin) / (sizeY-1)
            dR = (RMax-RMin) / (sizeZ-1)
        Then:
            M[i][j][k] = strength at position (thetaMin + i*dTheta, phiMin + j*dPhi, RMin + k*dR)
        Requires previous WalabotAPI.Trigger(). Provides data based on last completed
        triggered image. Output is details of array variable populated by
        provided image data. Provided image data is dependent on current
        configured arena and on current configuration from
        WalabotAPI.SetDynamicImageFilter() and WalabotAPI.SetThreshold().
        Returns:
            rasterImage:    Name of array variable populated by output image data
            sizeX:          Dimension of array variable populated by output
                            image data.
            sizeY:          Dimension of array variable populated by output
                            image data.
            sizeZ:          Dimension of array variable populated by output
                            image data.
            power:          Peak measured power in arena (value of strongest
                            pixel).
    """
    carr_rasterImage = POINTER(c_int)()
    sizeX = c_int(0)
    sizeY = c_int(0)
    sizeZ = c_int(0)
    power = c_double(0.0)

    res = _wlbt.Walabot_GetRawImage(byref(carr_rasterImage), byref(sizeX), byref(sizeY), byref(sizeZ), byref(power))
    _RaiseIfErr(res)

    X, Y, Z = sizeX.value, sizeY.value, sizeZ.value
    ## transform 1D C-style array into 3D python list.
    rasterImage=[[[ carr_rasterImage[z*X*Y+y*X+x] for z in range(Z)] for y in range(Y)] for x in range(X)]
    return rasterImage, X, Y, Z, power.value

    

def GetImageEnergy():
    """ Provides the sum of all the raw images pixels signal power.
        Requires previous Trigger(); provides data based on last completed
        triggered image. Provided image data is dependent on current
        configured arena.
        Returns:
            energy:         Number representing the sum of all the raw images
                            pixels signal power
    """
    energy = c_double()
    _RaiseIfErr(_wlbt.Walabot_GetImageEnergy(byref(energy)))
    return energy.value

class _Ctypes_ImagingTarget(Structure):
        _fields_ = [("type", c_int), ("angleDeg", c_double),
            ("xPosCm", c_double), ("yPosCm", c_double), ("zPosCm", c_double),
            ("widthCm", c_double), ("amplitude", c_double)]

def GetImagingTargets():
    """ Provides a list of identified targets.
        Available only if the short-range scan profile was used. Requires
        previous Trigger(); provides data based on last completed triggered
        image. Provided image data is dependent on current configured arena
        and on current configuration from SetDynamicImageFilter() and
        SetThreshold(). Note: In the current API version, provides only the
        single target with the strongest signal, in format appropriate for pipe
        Returns:
            targets:        List of targets (in current API version, a single
                            target) (of ImagingTarget namedtuple)
    """
    targets, numTargets = pointer(_Ctypes_ImagingTarget()), c_int()
    res = _wlbt.Walabot_GetImagingTargets(byref(targets), byref(numTargets))
    _RaiseIfErr(res)
    return [ImagingTarget(targets[i].type, targets[i].angleDeg,
        targets[i].xPosCm, targets[i].yPosCm, targets[i].zPosCm,
        targets[i].widthCm, targets[i].amplitude)
        for i in range(numTargets.value)]

class _Ctypes_SensorTarget(Structure):
        _fields_ = [('xPosCm', c_double), ('yPosCm', c_double),
            ('zPosCm', c_double), ('amplitude', c_double)]

def GetSensorTargets():
    """ Provides a list of and the number of identified targets.
        Available only if one of the Sensor scan profiles was used. Requires
        previous Trigger(); provides data based on last completed triggered
        image. Provided image data is dependent on current configured arena
        and on current configuration from SetDynamicImageFilter() and
        SetThreshold().
        Returns:
            targets:        List of targets (of SensorTarget namedtuple)
    """
    targets, numTargets, = pointer(_Ctypes_SensorTarget()), c_int()
    res = _wlbt.Walabot_GetSensorTargets(byref(targets), byref(numTargets))
    _RaiseIfErr(res)
    return [SensorTarget(targets[i].xPosCm, targets[i].yPosCm,
        targets[i].zPosCm, targets[i].amplitude)
        for i in range(numTargets.value)]

def SetDynamicImageFilter(filterType):
    """ Dynamic-imaging filter removes static signals, leaving only changing
        signals. Specify filter algorithm to use. Filter is not applied to
        GetImageEnergy(). To check the current value, use
        GetDynamicImageFilter().
        Args:
            filterTyp       Filter algorithm to use
    """
    _RaiseIfErr(_wlbt.Walabot_SetDynamicImageFilter(filterType))

def GetDynamicImageFilter():
    """ Obtains current Walabot Dynamic-imaging filter setting.
        Can be called at any time, default value is FILTER_TYPE_NONE
        Returns:
            filterType      Dynamic-imaging filter current setting.
    """
    filterType = c_int()
    _RaiseIfErr(_wlbt.Walabot_GetDynamicImageFilter(byref(filterType)))
    return filterType.value

def GetVersion():
    """ Obtains current Walabot version.
        The version is build from according to the following parameters:
        1) HW version - Walabot device revision.
        2) SW version - Walabot SW revision.
        3) Regulation information (where applicable).
        The function can be called only after connecting to the device.
        Returns:
            version:        Walabot version
    """
    version = c_char_p()
    _RaiseIfErr(_wlbt.Walabot_GetVersion(byref(version)))
    return version.value.decode('ascii')

# The linear permittivity of a homogeneous material is usually given relative to
# that of free space, as a relative permittivity epsilon*r. Valid values: {1-30}
PARAM_DIELECTRIC_CONSTANT = "DielectricConstant"
# Walabot internal pipe sensor detection percentage. Read only parameter.
PARAM_CONFIDENCE_FACTOR = "ConfidenceFactor"

def SetAdvancedParameter(paramName, value):
    """ Set advanced Walabot parameter.
        Parameters:
            paramName:      Advance parameter name, can be one of the following:
                            PARAM_CONFIDENCE_FACTOR, PARAM_DIELECTRIC_CONSTANT.
            value:          Value to set (floating-point value).
    """
    _RaiseIfErr(_wlbt.Walabot_SetAdvancedParameter(paramName.encode('ascii'),
        c_double(value)))

def GetAdvancedParameter(paramName):
    """ Obtains current Walabot advnaced parameter value.
        Args:
            paramName:      Advance parameter name, can be one of the following:
                            PARAM_CONFIDENCE_FACTOR, PARAM_DIELECTRIC_CONSTANT.
        Returns:
            value:          Current value of parameter (floating-point value).
    """
    value = c_double(0.0)
    _RaiseIfErr(_wlbt.Walabot_GetAdvancedParameter(paramName.encode('ascii'),
        byref(value)))
    return value.value
