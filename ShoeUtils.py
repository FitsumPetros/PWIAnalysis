import os
import pickle
import socket
import struct
import threading
import time
import tkinter as tk
import warnings
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.io as sio
import statsmodels.stats.weightstats as smws
from scipy import integrate
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# import Tools.PlotUtils as plotT


class smartShoe(object):
    """
    This object contains all the data collected by the shoes
    """

    def __init__(self, isLeft, keepAll=True, sizeBuf=1000):
        """
        Bool isLeft: if True shoe is treated as Left (Master)
        Bool keepAll: if True all values are kept as lists, else only a buffer of sizeBuf values are kept
        """
        self.keepAll = keepAll
        self.sizeBuf = sizeBuf
        self.binaryFile = bytes(0)
        self.fmt = '<3c I 13h 4c'
        # self.fmt = '=3c I 13h 4c'
        self.isLeft = isLeft
        self.pressDisplay = None
        self.plotP = None
        self.now = time.time()
        self.timestamp = []
        self.pToe = []
        self.pBall = []
        self.pHeel = []

        # acc
        self.ax = []
        self.ay = []
        self.az = []

        # gx
        self.gx = []
        self.gy = []
        self.gz = []

        # Euler
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.phase = []
        self.sync = []
        self.euAngles = [self.pitch,
                         self.yaw,
                         self.roll]
        self.EUorder = ['EUy', 'EUz', 'EUx']
        self.all = [self.pToe,
                    self.pBall,
                    self.pHeel,
                    self.ax,
                    self.ay,
                    self.az,
                    self.gx,
                    self.gy,
                    self.gz,
                    self.roll,
                    self.pitch,
                    self.yaw]
        self.names = ['pToe', 'pBall', 'pHeel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'EUy', 'EUz', 'EUx']
        # self.names = ['pToe', 'pBall', 'pHeel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz','EUx', 'EUy', 'EUz']
        self.namesWrong = ['pToe', 'pBall', 'pHeel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'EUx', 'EUy', 'EUz']
        self.normalValues = [[None, None] for _ in self.names]
        self.dataFrame = pandas.DataFrame(columns=self.names + ['sync'])
        self.syncFound = None
        # self.dataFrame = pandas.DataFrame(data=self.all, columns=self.names, index= self.timestamp)

    def resetShoe(self):
        self.now = time.time()
        self.timestamp = []
        self.pToe = []
        self.pBall = []
        self.pHeel = []

        # acc
        self.ax = []
        self.ay = []
        self.az = []

        # gx
        self.gx = []
        self.gy = []
        self.gz = []

        # Euler
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.phase = []
        self.sync = []
        self.euAngles = [self.pitch,
                         self.yaw,
                         self.roll]
        self.EUorder = ['EUy', 'EUz', 'EUx']
        self.all = [self.pToe,
                    self.pBall,
                    self.pHeel,
                    self.ax,
                    self.ay,
                    self.az,
                    self.gx,
                    self.gy,
                    self.gz,
                    self.roll,
                    self.pitch,
                    self.yaw]

        self.normalValues = [[None, None] for _ in self.names]
        self.dataFrame.drop(self.dataFrame.index, inplace=True)
        self.syncFound = None

    def updateVars(self):
        self.pToe = self.all[0]
        self.pBall = self.all[1]
        self.pHeel = self.all[2]
        self.ax = self.all[3]
        self.ay = self.all[4]
        self.az = self.all[5]
        self.gx = self.all[6]
        self.gy = self.all[7]
        self.gz = self.all[8]
        self.mx = self.all[9]
        self.my = self.all[10]
        self.mz = self.all[11]

    def saveBinaryFile(self, fileN):
        if self.isLeft:
            fileN += '_L.bin'
        else:
            fileN += '_R.bin'
        with open(fileN, 'wb') as f:
            f.write(self.binaryFile)

    def makeBinaryFileAgain(self):
        fmt = self.fmt
        fmt2 = fmt[1:] + ' '
        totalPacks = len(self.timestamp)
        allFmt = fmt + ' ' + fmt2 * (totalPacks - 1)
        if self.isLeft:
            side = 'l'
        else:
            side = 'r'
        # create all packet file
        pAllList = (
            [bytes([0x1]), bytes([0x2]), bytes([0x3])] + list(vals) + [side.encode()] + [bytes([0xA]), bytes([0xB]),
                                                                                         bytes([0xC])] for vals in
            zip(self.timestamp,
                self.pToe, self.pBall, self.pHeel,
                self.ax, self.ay, self.az,
                self.gx, self.gy, self.gz,
                self.roll, self.pitch, self.yaw,
                self.sync))
        pAll = [item for sublist in pAllList for item in sublist]
        self.binaryFile = struct.pack(allFmt, *pAll)

    def getPressVal(self, a):
        """
        :list a: list of pressure values
        :return: last registered pressure value of a
        """
        f = lambda x: (x[-1] - min(x)) / (max(x) - min(x))
        if len(a) == 0 or (max(a) - min(a)) == 0:
            return 0
        else:
            return f(a)

    def appendData(self, data, t, sync, append2list=False):
        if not append2list:
            self.dataFrame.loc[t, 'sync'] = sync
            self.dataFrame.loc[t, self.names] = data
            self.dataFrame.loc[t, ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'EUx', 'EUy', 'EUz']] *= 1.0 / 8000
            if not self.keepAll and self.dataFrame.shape[0] > self.sizeBuf:
                self.dataFrame.drop(self.dataFrame.index[0], inplace=True)
        else:
            if self.keepAll:
                self._appendDataAll(data, t, sync)
            else:
                self._appendDataBuff(data, t, sync)

    def _appendDataAll(self, data, t, sync):
        """
        :param data: list of values to append, they need to be in the same order as self.all
        :param t: timestamp when data was collected
        :param sync: 0 or 1 if a signal was received
        :return: nothing
        """
        self.timestamp.append(t)
        self.sync.append(sync)
        # print('%s,%s'%(len(data),len(self.all)))
        for d, v in zip(data, self.all):
            # print(d)
            v.append(d)

    def _appendDataBuff(self, data, t, sync):
        """
        :param data: list of values to append, they need to be in the same order as self.all
        :param t: timestamp when data was collected
        :param sync: 0 or 1 if a signal was received
        :return: nothing
        """
        self.timestamp.append(t)
        self.sync.append(sync)
        popEle = len(self.timestamp) > self.sizeBuf
        if popEle:
            self.timestamp.pop(0)
            self.sync.pop(0)
        # print('%s,%s'%(len(data),len(self.all)))
        for d, v in zip(data, self.all):
            # print(d)
            v.append(d)
            if popEle:
                v.pop(0)

    def makeMatrix(self, freq=100, cutAtSync=True, normal=False, filter=False, oldShoe=False):
        """
        :param useAngle:
        :param freq: desired frequency of resampling
        :param cutAtSync: if True, cuts the data to sync with sync signal
        :return: time[s]
                 newMat, matrix with all values collected at time[s] in the same order as self.all
        """
        # make time vector
        if self.dataFrame.size == 0:
            self.createPandasDataFrame()
        t = self.dataFrame.index.values
        # tNew = np.linspace(0, t[-1], (t[-1] - t[0]) * freq)
        # tNew = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * freq))
        tNew = np.linspace(t[0], t[-1], int(np.floor((t[-1] - t[0]) * freq)))
        mat = self.dataFrame.values.copy()
        if filter or normal:
            for i, (val, nVals) in enumerate(zip(self.all, self.normalValues)):
                if normal:
                    mat[:, i] = normTraj(mat[:, i], nVals[0], nVals[1])
                if filter:
                    mat[:, i] = butter_lowpass_filter(mat[:, i], 10, freq, order=4)
        # mat = np.empty((t.size, len(self.all)))
        # for i, val in enumerate(self.all):
        #     if normal:
        #         mat[:, i] = normTraj(np.array(val))
        #     else:
        #         mat[:, i] = np.array(val)
        #     if filter:
        #         mat[:, i] = butter_lowpass_filter(mat[:, i], 10, freq, order=2)
        newMat = reSampleMatrix(mat, 0, t=t, ti=tNew)
        if cutAtSync:
            s = np.where(np.array(self.sync) == 1)[0]
            if len(s) is 0:
                print('Sync not found')
                self.syncFound = False
                return tNew, newMat
            self.syncFound = True
            s = int(np.round(self.timestamp[s[0]] / 1000.0 * freq))
            newMat = newMat[s:, :]
            tNew = np.arange(len(newMat)) / freq
        if oldShoe:
            # names = ['pToe', 'pBall', 'pHeel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
            newMat = np.concatenate((newMat[:, 9:12], newMat[:, 3:6], newMat[:, 1:3]), 1)
            # newMat = np.concatenate((newMat[:, 9:12], newMat[:, 3:6], newMat[:, 0:1], newMat[:, 3:4]), 1)
        return tNew, newMat

    def complementaryFilter(self, acc, gyro, t, freq):
        # gyro needs to be in rads
        ang = np.zeros_like(gyro)
        ang[:, 0] = 0.98 * integrate.cumtrapz(gyro[:, 0], x=t, initial=0)
        ang[:, 1] = 0.98 * integrate.cumtrapz(gyro[:, 1], x=t, initial=0)
        ang[:, 2] = integrate.cumtrapz(gyro[:, 2], x=t, initial=0)

        accAngX = np.arctan2(acc[:, 1], acc[:, 2])
        accAngY = np.arctan2(-acc[:, 0], np.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2))

        ang[:, 0] += 0.02 * accAngX
        ang[:, 1] += 0.02 * accAngY
        for j in range(ang.shape[1]):
            ang[:, j] = butter_lowpass_filter(ang[:, j], 10, freq, order=2)
        return np.arccos(np.cos(ang))

    def createPandasDataFrame(self, useSync=False, returnObj=False, freq=None, oldShoe=False, filter=False):
        if len(self.ax) is 0:
            self.dataFrame = pandas.DataFrame(columns=self.names + ['sync'])
            warnings.warn("Shoe object is empty, hence dataFrame is empty", DeprecationWarning)
        else:
            if freq is not None or oldShoe:
                t, d = self.makeMatrix(freq=freq, cutAtSync=useSync, normal=False, filter=filter, oldShoe=oldShoe)
                if oldShoe:
                    self.names = ['EUx', 'EUy', 'EUz', 'ax', 'ay', 'az', 'pBall', 'pHeel']
                self.dataFrame = pandas.DataFrame(data=d,
                                                  columns=self.names,
                                                  index=t)
            else:
                d = np.array(self.all).T
                t = np.array(self.timestamp) / 1000
                tS = 0
                if useSync:
                    s = np.where(np.array(self.sync) == 1)[0]
                    if len(s) is 0:
                        warnings.warn('Sync not found')
                        self.syncFound = False
                        tS = self.timestamp[0]
                    else:
                        tS = self.timestamp[s[0]]
                        d = d[s, :]
                        t = t[s]
                        self.syncFound = True
                # divide by 8000, 8000 is the scale factor we are using for

                self.dataFrame = pandas.DataFrame(data=d,
                                                  columns=self.namesWrong,
                                                  index=(t - tS))
                self.dataFrame.loc[:, ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'EUx', 'EUy', 'EUz']] *= 1.0 / 8000

                self.dataFrame = self.dataFrame[self.names]

            # else:
            # self.dataFrame.index /= 1000
            # self.dataFrame.loc[:, ['mx', 'my', 'mz']] = np.arcsin(np.sin(self.dataFrame.loc[:, ['mx', 'my', 'mz']]))
        if returnObj:
            return self.dataFrame.copy()

    def plotValue(self, valNames, **kwargs):
        if self.dataFrame is None:
            self.createPandasDataFrame()
        if type(valNames) is not list:
            valNames = [valNames]
        for valName in valNames:
            if valName not in self.names:
                raise ValueError(
                    'Desired sensor not in shoe, received: %s, valid names are: %s' % (valName, self.names))

            self.dataFrame.plot(y=valName, **kwargs)

    def createListsFromBinary(self):
        bytes_read = self.binaryFile
        fmt2 = self.fmt[1:] + ' '
        fileSize = len(bytes_read)
        packetSize = struct.calcsize(self.fmt)
        totalPacks = fileSize // packetSize
        allFmt = self.fmt + ' ' + fmt2 * (totalPacks - 1)
        pAll = struct.unpack(allFmt, bytes_read)
        nVariables = len(self.all)
        # divide the list into chunks of nVariables, i.e. samples
        self.timestamp = [pAll[i + 3] for i in range(0, len(pAll), nVariables + 1)]
        self.sync = [pAll[i + 16] for i in range(0, len(pAll), nVariables + 1)]
        for j in range(len(self.all)):
            self.all[j][:] = [pAll[i + j + 4] for i in range(0, len(pAll), nVariables + 1)]


class DeepSoleObject(smartShoe):
    def __init__(self, ID='', **kwargs):
        super(DeepSoleObject, self).__init__(True, **kwargs)
        self.ID = ID

    def saveBinaryFile(self, fileN):
        fileN += '_' + self.ID + '.bin'
        with open(fileN, 'wb') as f:
            f.write(self.binaryFile)

    def makeBinaryFileAgain(self):
        fmt = self.fmt
        fmt2 = fmt[1:] + ' '
        totalPacks = len(self.timestamp)
        allFmt = fmt + ' ' + fmt2 * (totalPacks - 1)
        side = self.ID
        # create all packet file
        pAllList = (
            [bytes([0x1]), bytes([0x2]), bytes([0x3])] + list(vals) + [side.encode()] + [bytes([0xA]), bytes([0xB]),
                                                                                         bytes([0xC])] for vals in
            zip(self.timestamp,
                self.pToe, self.pBall, self.pHeel,
                self.ax, self.ay, self.az,
                self.gx, self.gy, self.gz,
                self.roll, self.pitch, self.yaw,
                self.sync))
        pAll = [item for sublist in pAllList for item in sublist]
        self.binaryFile = struct.pack(allFmt, *pAll)


class MatObject(object):
    def __init__(self, fName=None, sheetname=0, freq=200, newMat=False):
        """

        :param fName: File name from zeno walkway
        :param sheetname: index of sheet
        """
        if newMat:
            self.raw = matFile2Pandas(fileN=fName, sheetname=sheetname)
        else:
            self.raw = matFile2PandasOld(file_name=fName, sheetname=sheetname)

        self.newMat = newMat
        self.freq = freq
        self.makeY()
        # if newMat:
        #     t, yR, yL, self.lap, self.labels, firstC, lastC = createPlotsFromMatForNew(d=self.raw, freq=freq)
        # else:
        #     t, yR, yL, self.lap, self.labels, firstC, lastC = createPlotsFromMat(d=self.raw, freq=freq)
        # self.t = t
        # self.yR = yR
        # self.yL = yL
        # self.binaryFunctions = pandas.DataFrame(data=np.stack((yR, yL), axis=1), index=t, columns=['R', 'L'])

    def makeYpercent(self):
        freq = self.freq
        if self.newMat:
            t, yR, yL, self.lap, self.labels, firstC, lastC = createPlotsFromMatForNew(d=self.raw, freq=freq,
                                                                                       usePercentage=True)
            _, yR2, yL2, self.lap, self.labels, firstC, lastC = createPlotsFromMatForNew(d=self.raw, freq=freq,
                                                                                         usePercentage=False)
        else:
            t, yR, yL, self.lap, self.labels, firstC, lastC = createPlotsFromMat(d=self.raw, freq=freq,
                                                                                 usePercentage=True)
            _, yR2, yL2, self.lap, self.labels, firstC, lastC = createPlotsFromMat(d=self.raw, freq=freq,
                                                                                   usePercentage=False)
        self.t = t
        self.yR = yR
        self.yL = yL
        self.phase = np.zeros_like(t) * np.nan
        # -1 is unknown
        # 0 is single support
        # 1 is double support
        nYr = np.logical_not(yR2.astype(np.bool))
        nYl = np.logical_not(yL2.astype(np.bool))
        # single support is not(yR) or not(yL
        m = np.logical_or(nYr, nYl)
        self.phase[m] = 0
        # double support is not(yR) and not(yL)
        m = np.logical_and(nYr, nYl)
        self.phase[m] = 1
        self.binaryFunctions = pandas.DataFrame(data=np.stack((yR, yL, self.phase), axis=1), index=t,
                                                columns=['R', 'L', 'phase'])

    def makeY(self):
        freq = self.freq
        if self.newMat:
            t, yR, yL, self.lap, self.labels, firstC, lastC = createPlotsFromMatForNew(d=self.raw, freq=freq,
                                                                                       usePercentage=False)
        else:
            t, yR, yL, self.lap, self.labels, firstC, lastC = createPlotsFromMat(d=self.raw, freq=freq,
                                                                                 usePercentage=False)
        self.t = t
        self.yR = yR
        self.yL = yL
        self.phase = np.zeros_like(t) * np.nan
        # -1 is unknown
        # 0 is single support
        # 1 is double support
        nYr = np.logical_not(yR.astype(np.bool))
        nYl = np.logical_not(yL.astype(np.bool))
        # single support is not(yR) or not(yL
        m = np.logical_or(nYr, nYl)
        self.phase[m] = 0
        # double support is not(yR) and not(yL)
        m = np.logical_and(nYr, nYl)
        self.phase[m] = 1
        self.binaryFunctions = pandas.DataFrame(data=np.stack((yR, yL, self.phase), axis=1), index=t,
                                                columns=['R', 'L', 'phase'])
        a = 1


class ShoeSubject(object):
    def __init__(self, leftShoe: smartShoe, rightShoe: smartShoe, subID: int, testType: str = 'Sub',
                 matInfo: MatObject = None, cutSync=False):
        """
        Create a subject of shoe recording
        :param leftShoe: object for left shoe
        :param rightShoe: object for right shoe
        :param subID: ID of the subject
        :param testType: Test name
        :param matInfo: Mat object with the data recorded from Zeno walkway
        """
        self.leftShoe = leftShoe
        self.rightShoe = rightShoe
        self.subID = subID
        self.matInfo = matInfo
        self.testType = testType
        self.myName = '%s_%03d' % (self.testType, self.subID)
        self.binaryFunctionTh = None
        self.binaryFunctionML = None
        self.cutSync = cutSync
        self.db = {}

    def createBinaryFunctionWithThreshold(self, th, freq=100, cutAtSync=True, filter=False):
        tR, mR = self.rightShoe.makeMatrix(freq=freq, cutAtSync=cutAtSync, normal=False, filter=filter)
        tL, mL = self.leftShoe.makeMatrix(freq=freq, cutAtSync=cutAtSync, normal=False, filter=filter)
        yR = binaryFromSensors([mR[:, 0], mR[:, 1], mR[:, 2]], th)
        yL = binaryFromSensors([mL[:, 0], mL[:, 1], mL[:, 2]], th)
        if yR.size > yL.size:
            s = yL.size
            yR = yR[:s]
            t = tL
        else:
            s = yR.size
            yL = yL[:s]
            t = tR
        self.binaryFunctionTh = pandas.DataFrame(data=np.stack((yR, yL), axis=1), index=t, columns=['R', 'L'])

    def savePickle(self, fName=None):
        if fName is None:
            root = tk.Tk()
            fName = tk.filedialog.asksaveasfilename(parent=root, title='Pickle file',
                                                    filetypes=(("pickle files", "*.pickle"), ("all files", "*.*")),
                                                    defaultextension='.pickle')
            root.withdraw()
            if fName is None:
                return 0
        with open(fName, 'wb') as f:
            pickle.dump(self, f)

    def saveMatlab(self, fName=None):
        if fName is None:
            root = tk.Tk()
            fName = tk.filedialog.asksaveasfilename(parent=root, title='Matlab file',
                                                    filetypes=(
                                                        ("Matlab files", "*.mat"), ("all files", "*.*")))
            root.withdraw()
            if fName is None:
                return 0
        lShoe = self.leftShoe.dataFrame.copy()
        rShoe = self.rightShoe.dataFrame.copy()
        if self.matInfo is None:
            sub = {"lShoe": lShoe.values, "rShoe": rShoe.values, "ShoeCols": lShoe.columns.values,
                   "lTime": lShoe.index.values, "rTime": rShoe.index.values}
        else:
            sub = {"lShoe": lShoe.values, "rShoe": rShoe.values, "ShoeCols": lShoe.columns.values,
                   "lTime": lShoe.index.values, "rTime": rShoe.index.values,
                   "MatYs": self.matInfo.binaryFunctions.values, "MatRaw": self.matInfo.raw.values,
                   "MatTime": self.matInfo.raw.index.values, "MatColumns": self.matInfo.raw.columns.values}

        sio.savemat(fName, sub)

    def saveExcel(self, fName=None):
        if fName is None:
            root = tk.Tk()
            fName = tk.filedialog.asksaveasfilename(parent=root, title='Save data as',
                                                    filetypes=[("Excel files", ("*.xls", "*.xlsx"))],
                                                    defaultextension='.xlsx')
            root.withdraw()
            if fName is '':
                return 0
        print(fName)
        lShoe = self.leftShoe.createPandasDataFrame(returnObj=True, useSync=self.cutSync)
        rShoe = self.rightShoe.createPandasDataFrame(returnObj=True, useSync=self.cutSync)
        writer = pandas.ExcelWriter(fName, engine='xlsxwriter')
        lShoe.to_excel(writer, sheet_name='Left Shoe')
        rShoe.to_excel(writer, sheet_name='Right Shoe')
        writer.save()
        writer.close()

    def make2Ddatabase(self, params=['Stride Length (cm.)'], freq=100.0, maxTime=None, maxStrideTime=None):

        if type(self.matInfo) is not MatObject:
            warnings.warn('Mat object is not a valid object')
        self.rightShoe.createPandasDataFrame(useSync=self.cutSync, freq=freq)
        self.leftShoe.createPandasDataFrame(useSync=self.cutSync, freq=freq)
        x = []
        xLen = []
        y = []
        for j, row in self.matInfo.raw.iterrows():
            a = row[params].values
            if row[params].valid().shape[0] is 0:
                continue

            st = int(row['First Contact (sec.)'] * freq)
            en = int(row['Last Contact (sec.)'] * freq)
            isLeft = row['isLeft']
            if isLeft:
                s = self.leftShoe
            else:
                s = self.rightShoe

            if st > s.dataFrame.shape[0] or en > s.dataFrame.shape[0]:
                continue
            if maxStrideTime is not None and en - st > maxStrideTime * freq:
                continue
            x.append(s.dataFrame.values[st:en])
            xLen.append(x[-1].shape[0])
            y.append(row[params].values)

        y1 = pandas.DataFrame(data=y, columns=params)
        if maxTime is None:
            maxTime = max(xLen)
        x1 = np.zeros((len(x), maxTime, x[0].shape[1]))
        for j, (xx, xL) in enumerate(zip(x, xLen)):
            x1[j, :xL] = xx.copy()
        m = np.logical_not(np.isnan(y1)).squeeze()
        self.db['x'] = x1[m]
        self.db['y'] = y1.values[m]

        a = 1


def rotFromVec(v):
    y = v / np.linalg.norm(v)
    x = np.cross(np.array([1, 0, 0]), y)
    z = np.cross(x, y)
    R = np.stack((x, y, z), axis=1)
    return R.T


def body2spaceEuler(v, eu, rotationOrder=[1, 2, 3], freeze=False, leftHanded=False):
    vNew = np.zeros_like(v)
    if len(vNew.shape) == 1:
        vNew = np.expand_dims(vNew, 1)
    if freeze:
        Rfre = euler2Rot(eu[0, 0], eu[0, 1], eu[0, 2], rotationOrder=rotationOrder).T
    for j in range(vNew.shape[0]):
        R = euler2Rot(eu[j, 0], eu[j, 1], eu[j, 2], rotationOrder=rotationOrder)
        if leftHanded:
            if freeze:
                vNew[j, :] = np.dot(v[j, :].T, np.dot(Rfre, R))
            else:
                # vNew[j, :] = np.dot(v[j, :].T, R)
                vNew[j, :] = np.dot(R.T, v[j, :])
        else:
            if freeze:
                vNew[j, :] = np.dot(Rfre, np.dot(R, v[j, :]))
            else:
                # vNew[j, :] = np.dot(R, v[j, :])
                R1 = np.array([[0, 0, 1],
                               [0, 1, 0],
                               [1, 0, 0]])
                vNew[j, :] = np.dot(R, np.dot(R1, v[j, :]))
    return vNew


def axisRotationMatrix(theta, axis):
    s = np.sin(theta)
    c = np.cos(theta)
    if axis is 1:
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
    elif axis is 2:
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    elif axis is 3:
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
    else:
        R = np.identity(3)

    return R


def euler2Rot(r1, r2, r3, rotationOrder=[1, 2, 3]):
    angles = [r1, r2, r3]
    # s1 = np.sin(x)
    # s2 = np.sin(y)
    # s3 = np.sin(z)
    # c1 = np.cos(x)
    # c2 = np.cos(y)
    # c3 = np.cos(z)
    # R = np.empty((3, 3))
    # R[0, 0] = s1 * s2 * s3 + c3 * c1
    # R[1, 0] = c2 * s3
    # R[2, 0] = c1 * s2 * s3 - c3 * s1
    #
    # R[0, 1] = s1 * s2 * c3 - s3 * c1
    # R[1, 1] = c2 * c3
    # R[2, 1] = c1 * s2 * c3 + s3 * s1
    #
    # R[0, 2] = s1 * c2
    # R[1, 2] = -1.0 * s2
    # R[2, 2] = c1 * c2

    R = np.identity(3)
    for j, a in zip(rotationOrder, angles):
        Rs = axisRotationMatrix(a, j)
        R = np.matmul(R, Rs)
    return R


def binaryFromSensors(signals, th):
    if type(signals) is not list:
        signals = [signals]
    bins = np.ones_like(signals[0], np.bool)
    for s in signals:
        bins = np.logical_and(np.greater_equal(normTraj(s), th), bins)
    return bins.astype(np.int)


def normTraj(x, mi=None, ma=None):
    """
    returns a normalized (min val = 0, max val = 1) version of x
    :param x: input trajectory
    :return: new x
    """
    xSo = np.sort(x)
    tenP = int(x.size * 0.1)
    if mi is None:
        mi = np.mean(xSo[:tenP])
    if ma is None:
        ma = np.mean(xSo[-tenP:])
    newX = (x.copy() - mi) / (ma - mi)
    return newX


def binaryFile2python(fileN=None, shareTime=True, fmt=None, V=2, readAll=True) -> (smartShoe, smartShoe):
    """
    :param fileN: path of binary file recorded by the shoes
    :param shareTime: if True, both shoes use the time from Master (this is desired)
    :return: right and left shoe objects
    """
    if fileN is None:
        root = tk.Tk()
        #
        fileN = tk.filedialog.askopenfilename(parent=root, title='Select Shoe binary file',
                                              filetypes=(("Binary files", "*.bin"), ("all files", "*.*")))
        root.withdraw()
        if fileN is None:
            return 0
    file = open(fileN, 'rb')
    l = smartShoe(True)
    r = smartShoe(False)
    if fmt is None:
        if V is 1:
            fmt = '!3c I 13h I 12h 3c'
            nVariables = 64
        else:
            fmt = '<3c I 13h 4c'
            nVariables = 20
    if readAll and V is not 1:
        fmt2 = fmt[1:] + ' '
        fileSize = os.path.getsize(file.name)
        packetSize = struct.calcsize(fmt)
        totalPacks = fileSize // packetSize
        allFmt = fmt + ' ' + fmt2 * (totalPacks - 1)
        # read all the file at 1
        bytes_read = file.read(fileSize)
        file.close()
        pAll = struct.unpack(allFmt, bytes_read)
        # divide the list into chunks of nVariables, i.e. samples
        s = smartShoe(True)
        side = np.array([pAll[i + 17] for i in range(0, len(pAll), nVariables + 1)])
        if not np.all(side == side[0]):
            warnings.warn('All packets not from same side')
        side = side[0]
        s.timestamp = [pAll[i + 3] for i in range(0, len(pAll), nVariables + 1)]
        s.sync = [pAll[i + 16] for i in range(0, len(pAll), nVariables + 1)]
        for j in range(len(s.all)):
            s.all[j][:] = [pAll[i + j + 4] for i in range(0, len(pAll), nVariables + 1)]
        # s.updateVars()
        if side.decode() is 'l':
            l = s
        else:
            s.isLeft = False
            r = s
        # pDivided = (pAll[i:i + nVariables] for i in range(0, len(pAll), nVariables + 1))
        # # p is totalPacks * 20
        # for p in pDivided:
        #     if p[17].decode() is 'l':
        #         l.binaryFile += bytes_read
        #         l.appendData(p[4:16], p[3], p[16])
        #     else:
        #         r.binaryFile += bytes_read
        #         r.appendData(p[4:16], p[3], p[16])

    else:
        if V is 1:
            # if fmt is None:
            #     fmt = '!3c I 13h I 12h 3c'
            # file = open(f, "rb")
            try:
                sizePack = struct.calcsize(fmt)
                bytes_read = file.read(sizePack)
                while bytes_read:
                    if len(bytes_read) != sizePack:
                        warnings.warn('File size not consistent')
                        bytes_read = file.read(struct.calcsize(fmt))
                        continue
                    p = struct.unpack(fmt, bytes_read)
                    # if p[0] == chr(1) and p[1] == chr(2) and p[2] == chr(3) and p[-3] == chr(0xA) and p[-2] == chr(0xB) and
                    #  p[-1] == chr(0xC):
                    if shareTime:
                        l.appendData(p[4:16], p[3], p[16])
                        r.appendData(p[18:30], p[3], p[16])
                    else:
                        l.appendData(p[4:16], p[3], p[16])
                        r.appendData(p[18:30], p[17], p[16])
                    bytes_read = file.read(struct.calcsize(fmt))
            finally:
                file.close()
        else:
            # if fmt is None:
            #     fmt = '<3c I 13h 4c'
            # print(len(unparsed))
            try:
                bytes_read = file.read(struct.calcsize(fmt))
                while bytes_read:
                    p = struct.unpack(fmt, bytes_read)
                    # print(int(p[17],8))
                    if p[17].decode() is 'l':
                        l.binaryFile += bytes_read
                        l.appendData(p[4:16], p[3], p[16])
                    else:
                        r.binaryFile += bytes_read
                        r.appendData(p[4:16], p[3], p[16])
                    bytes_read = file.read(struct.calcsize(fmt))
            finally:
                file.close()
    # fSave = tk.filedialog.asksaveasfile()
    # r.createPandasDataFrame()
    # l.createPandasDataFrame()
    shoes = [r, l]
    # if fSave == None:
    #    return shoes
    # np.save(fSave,shoes)
    return shoes


def readBinaryFile2Pandas(fmt, fileName=None, colNames=None, header=None, footer=None) -> pandas.DataFrame:
    """

    :param fmt: String with a struct format
    :type fmt: str
    :param fileName: Binary file path. If is None, an open file dialog is opened
    :type fileName: str
    :param colNames: (optional) column names in a list
    :type colNames: str
    :param header: (optional) header characters
    :type header: str
    :param footer: (optional) footer characters
    :type footer: str
    :return: returns a pandas dataFrame with the data
    """
    if fileName is None:
        root = tk.Tk()
        #
        fileName = tk.filedialog.askopenfilename(parent=root, title='Select Shoe binary file',
                                                 filetypes=(("Binary files", "*.bin"), ("all files", "*.*")))
        root.withdraw()
        if fileName is None:
            return 0
    file = open(fileName, 'rb')
    fmt2 = fmt[1:] + ' '
    fileSize = os.path.getsize(file.name)
    packetSize = struct.calcsize(fmt)
    totalPacks = fileSize // packetSize
    allFmt = fmt + ' ' + fmt2 * (totalPacks - 1)
    # read all the file at 1
    bytes_read = file.read(fileSize)
    file.close()
    pAll = struct.unpack(allFmt, bytes_read)
    nVars = len(struct.unpack(fmt, bytes(struct.calcsize(fmt))))

    if header is None and footer is None:
        data = np.array(pAll).reshape((-1, nVars))
        header = []
        footer = []
    else:
        if header is not None:
            headerVals = np.array([pAll[i: i + len(header)] for i in range(0, len(pAll), nVars)])
            if not np.all(headerVals == header):
                warnings.warn('Packets corrupted!')
        else:
            header = []

        if footer is not None:
            footerVals = np.array([pAll[i+nVars-len(footer):i+nVars] for i in range(0, len(pAll), nVars)])
            if not np.all(footerVals == footer):
                warnings.warn('Packets corrupted!')
        else:
            footer = []

        data = np.array([pAll[i + len(header): i + nVars - len(footer)] for i in range(0, len(pAll), nVars)])

    if colNames is None:
        colNames = ['var%d' % j for j in range(nVars - len(header) - len(footer))]

    return pandas.DataFrame(data=data, columns=colNames)



# UDPs
class UDPsend(object):
    """
    Object for sending data through UDP
    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP

    def send(self, msg, addr=('192.168.42.1', 12354)):
        """
        :param msg: message to send
        :param addr: tuple of length 2, (ip address, port number)
        :return:
        """
        self.sock.sendto(msg, addr)

    def closeSocket(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

    def reOpenSocket(self):
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP


class UDPreceiveNoThread(object):
    def __init__(self, portN, parseFn):
        """
        :param leftShoe: left shoe object
        :param rightShoe: right shoe object
        :param console: tkInter text to be used as "console"
        :param portN: port number for receiving
        """
        self.data = ""
        self.dataRead = False
        self.t1 = None
        self.portN = portN
        self.parseFn = parseFn
        # self.reOpenSocket()

    def reOpenSocket(self):
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP
        self.sock.bind(("", self.portN))

    def receiveData(self):
        """
        Main function for receiving, this is usually on a different thread
        :param e:
        :return:
        """
        self.reOpenSocket()
        while True:
            data, addr = self.sock.recvfrom(1024)
            if data is not None:
                self.parseFn(data, addr)
        print("Thread ended")


class UDPreceiveProto(object):

    def __init__(self, console, portN=12345, bufferedData=None):
        """
        :param console: tkInter text to be used as "console"
        :param portN: port number for receiving
        """

        self.data = ""
        self.dataRead = False
        self.t1 = None
        self.rec = True
        self.console = console
        self.portN = portN
        if bufferedData is None:
            self.bufferedData = []
        else:
            self.bufferedData = bufferedData

    def reOpenSocket(self):
        self.sock = socket.socket(socket.AF_INET,  # Internet
                                  socket.SOCK_DGRAM)  # UDP
        self.sock.bind(("", self.portN))

    def _parseData(self, unparsed):
        """
        Parses network data and process it accordingly
        :param unparsed:
        :return:
        """
        print(unparsed)
        self.console.set(unparsed.decode("utf-8", "ignore"))

    def __receiveData(self, e):
        """
        Main function for receiving, this is usually on a different thread
        :param e:
        :return:
        """
        while e.is_set():
            data, addr = self.sock.recvfrom(1024)
            if data is not None:
                self._parseData(data)
        print("Thread ended")

    def clearBuffer(self):
        self.bufferedData = []

    def startThread(self):
        """
        Starts the receiving thread
        :return:
        """
        self.rec = True
        self.reOpenSocket()
        self.rec1 = threading.Event()
        self.rec1.set()
        self.t1 = threading.Thread(target=self.__receiveData, args=[self.rec1])
        self.t1.daemon = True
        self.t1.start()

    def stopThread(self):
        self.rec = False
        self.rec1.clear()
        time.sleep(0.5)
        self.closeSocket()

    def closeSocket(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


class UDPreceiveAllObjects(UDPreceiveProto):
    def __init__(self, console, soleObjects, portN=12345, bufferedData=None):
        super(UDPreceiveAllObjects, self).__init__(console, portN, bufferedData)
        self.soleObjects = soleObjects
        self.fmt = DeepSoleObject().fmt

    def _parseData(self, unparsed):
        if unparsed[0] == 1 and unparsed[1] == 2 and unparsed[2] == 3:
            # 3 char for open, uint64 for timestamp, 24 short for values and 3 char for close
            # fmt = '=3c I 13H 4c'
            fmt = self.fmt
            p = struct.unpack(fmt, unparsed)
            k = p[17].decode()
            if not k in [kk for kk in self.soleObjects.keys()]:
                warnings.warn('%s not in objects' % k)
                return
            # print(len(unparsed))
            self.soleObjects[k]['shoe'].binaryFile += unparsed
            self.soleObjects[k]['shoe'].appendData(p[4:16], p[3], p[16])

        else:
            print(unparsed)
            self.console.set(unparsed.decode("utf-8", "ignore"))


class UDPreceive(UDPreceiveProto):
    """
    Object to receive data through UDP
    """

    def __init__(self, leftShoe, rightShoe, console, portN=12345, ver=1, bufferedData=None):
        """
        :param leftShoe: left shoe object
        :param rightShoe: right shoe object
        :param console: tkInter text to be used as "console"
        :param portN: port number for receiving
        """
        super(UDPreceive, self).__init__(console, portN, bufferedData)
        self.leftShoe = leftShoe
        self.rightShoe = rightShoe
        self.ver = ver

    def _parseData(self, unparsed):
        """
        Parses network data and process it accordingly
        :param unparsed:
        :return:
        """
        if self.ver is 1:
            self._parseV1(unparsed)
        else:
            self._parseV2(unparsed)

    def _parseV1(self, unparsed):
        """
                Parses network data and process it accordingly
                :param unparsed:
                :return:
                """
        if unparsed[0] == 1 and unparsed[1] == 2 and unparsed[2] == 3:
            # 3 char for open, uint64 for timestamp, 24 short for values and 3 char for close
            fmt = '!3c I 13H I 12H 3c'
            # print(len(unparsed))
            p = struct.unpack(fmt, unparsed)
            self.leftShoe.appendData(p[4:16], p[3], p[16])
            self.rightShoe.appendData(p[18:30], p[17], p[16])
        elif unparsed[0] == 0xF and unparsed[1] == 0xF and unparsed[2] == 0xA:
            # 3 char for open, uint64 for timestamp, 24 short for values and 3 char for close
            fmt = '=3c 3H 4c'
            p = struct.unpack(fmt, unparsed)
        else:
            print(unparsed)
            self.console.set(unparsed.decode("utf-8"))

    def _parseV2(self, unparsed):
        """
                Parses network data and process it accordingly
                :param unparsed:
                :return:
                """
        if unparsed[0] == 1 and unparsed[1] == 2 and unparsed[2] == 3:
            # 3 char for open, uint64 for timestamp, 24 short for values and 3 char for close
            # fmt = '=3c I 13H 4c'
            fmt = self.rightShoe.fmt
            # print(len(unparsed))
            p = struct.unpack(fmt, unparsed)
            # print(p[17].decode())
            if p[17].decode() is 'l':
                self.leftShoe.binaryFile += unparsed
                self.leftShoe.appendData(p[4:16], p[3], p[16])
            else:
                self.rightShoe.binaryFile += unparsed
                self.rightShoe.appendData(p[4:16], p[3], p[16])
        elif unparsed[0] == 0xF and unparsed[1] == 0xF and unparsed[2] == 0xA:
            # 3 char for open, uint64 for timestamp, 24 short for values and 3 char for close
            fmt = '=3c 3H 4c'
            p = struct.unpack(fmt, unparsed)
        else:
            print(unparsed)
            self.console.set(unparsed.decode("utf-8", "ignore"))
            # self.bufferedData.append(unparsed)

def is_not_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    modified_z_score = points.copy()
    p = np.isnan(points)
    if np.any(p):
        points = points[~p]
        modified_z_score[p] = thresh * 100
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score[~p] = 0.6745 * diff / med_abs_deviation

    return modified_z_score < thresh


def removeOutlierFromPandas(df, inplace=False, replaceVal=None, verbose=False, sigmas=3, col=None):

    if col is None:
        tot = np.abs(df - df.mean()) >= (sigmas * df.std())
        if verbose:
            print('Removed %d values of %d, total %f%%' % (
                np.sum(tot.values), tot.size, np.sum(tot.values) / tot.size * 100.0))
        if inplace:
            if replaceVal is None:
                df = df[~tot]
            else:
                df[tot] = replaceVal
        else:
            if replaceVal is None:
                return df[~tot].copy()
            else:
                aux = df.copy()
                aux[tot] = replaceVal
                return aux
    else:
        ddf = df[col]
        tot = ~is_not_outlier(ddf.values, sigmas)
        if verbose:
            print('Removed %d values of %d, total %f%%' % (np.sum(tot), tot.size, np.sum(tot) / tot.size * 100.0))
        if inplace:
            if replaceVal is None:
                df = df.loc[~tot, :]
            else:
                df[tot] = replaceVal
        else:
            if replaceVal is None:
                return df[~tot, :].copy()
            else:
                aux = df.copy()
                aux[tot] = replaceVal
                return aux


def matFile2Pandas(fileN=None, sheetname=0, flipSide=False):
    if fileN is None:
        root = tk.Tk()
        #
        fileN = tk.filedialog.askopenfilename(parent=root, title='Select Mat Excel file',
                                              filetypes=(("Excel files", ("*.xls", "*.xlsx")), ("all files", "*.*")))
        root.withdraw()
        if fileN is None:
            return 0
    # names = ["num", "Index", "First Contact (sec.)", "Last Contact (sec.)", "Integ. Pressure (p x sec.)",
    #          "Foot Length (cm.)", "Foot Length %", "Foot Width (cm.)", "Foot Area (cm. x cm.)", "Foot Angle (degrees)",
    #          "Toe In/Out Angle (degrees)", "Foot Toe X Location (cm.)", "Foot Toe Y Location (cm.)",
    #          "Foot Heel X Location (cm.)", "Foot Heel Y Location (cm.)", "Foot Center X Location (cm.)",
    #          "Foot Center Y Location (cm.)", "Step Length (cm.)", "Absolute Step Length (cm.)", "Stride Length (cm.)",
    #          "Stride Width (cm.)", "Step Time (sec.)", "Stride Time (sec.)", "Stride Velocity (cm./sec.)",
    #          "DOP (degrees)", "Gait Cycle Time (sec.)", "Stance Time (sec.)", "Stance %", "Swing Time (sec.)",
    #          "Swing %", "Single Support (sec.)", "Single Support %", "Initial D. Support (sec.)",
    #          "Initial D. Support %", "Terminal D. Support (sec.)", "Terminal D. Support %", "Total D. Support (sec.)",
    #          "Total D. Support %", "CISP Time (sec.)", "CISP AP (%)", "CISP ML (%)", "Stance COP Dist. (cm.)",
    #          "SS COP Dist. (cm.)", "DS COP Dist. (cm.)", "Stance COP Dist. %", "SS COP Dist. %",
    #          "Stance COP Path Eff. %", "SS COP Path Eff. %", "DS COP Path Eff.%"]

    s = pandas.read_excel(fileN, sheet_name=sheetname, header=None)
    colA = s.iloc[:, 0]
    # find headers
    names = s.iloc[np.where(colA == '#Samples')[0][0] - 1, :].values
    names[0] = "num"
    names[1] = "Index"
    st = np.where(colA == 1)[0][0]
    d = pandas.DataFrame(data=s.iloc[st:, :].values, columns=names)
    ind = d['Index']
    laps = [int(x.split(':')[0]) for x in ind]
    stInd = [int(x.split(':')[1].split(' ')[2]) for x in ind]
    sides = ['Left' in x.split(':')[1] for x in ind]
    d['Laps'] = laps
    if flipSide:
        d['isLeft'] = [not s for s in sides]
    else:
        d['isLeft'] = sides
    d['StepIndex'] = stInd
    newNames = np.concatenate((['Laps', 'isLeft', 'StepIndex'], names))
    d = d[newNames]
    d.iloc[:, 5:] = d.iloc[:, 5:].astype(np.float)
    return d


def matFile2PandasOld(file_name=None, sheetname=0):
    # Note: for new version mat, the excel may be different from the old version, please check
    names = ["num", "Index", "First Contact (sec.)", "Last Contact (sec.)", "Integ. Pressure (p x sec.)",
             "Foot Length (cm.)", "Foot Length %", "Foot Width (cm.)", "Foot Area (cm. x cm.)", "Foot Angle (degrees)",
             "Toe In/Out Angle (degrees)", "Foot Toe X Location (cm.)", "Foot Toe Y Location (cm.)",
             "Foot Heel X Location (cm.)", "Foot Heel Y Location (cm.)", "Foot Center X Location (cm.)",
             "Foot Center Y Location (cm.)", "Step Length (cm.)", "Absolute Step Length (cm.)", "Stride Length (cm.)",
             "Stride Width (cm.)", "Step Time (sec.)", "Stride Time (sec.)", "Stride Velocity (cm./sec.)",
             "DOP (degrees)", "Gait Cycle Time (sec.)", "Stance Time (sec.)", "Stance %", "Swing Time (sec.)",
             "Swing %", "Single Support (sec.)", "Single Support %", "Initial D. Support (sec.)",
             "Initial D. Support %", "Terminal D. Support (sec.)", "Terminal D. Support %", "Total D. Support (sec.)",
             "Total D. Support %", "CISP Time (sec.)", "CISP AP (%)", "CISP ML (%)", "Stance COP Dist. (cm.)",
             "SS COP Dist. (cm.)", "DS COP Dist. (cm.)", "Stance COP Dist. %", "SS COP Dist. %",
             "Stance COP Path Eff. %", "SS COP Path Eff. %", "DS COP Path Eff.%"]
    d = pandas.read_excel(file_name, sheet_name=sheetname, header=None, names=names, skiprows=20,
                          usecols=len(names) - 1)
    # usecols if int then indicates last column to be parsed
    return d


def createPlotsFromMatForNew(d=None, f=None, freq=100, sheetname=0, usePercentage=False):
    """
    Creates signals representing foot on the groung or not on the ground
    :param d: Pandas Dataframe obtained from function
    :param f: File name of the excel file processed with pkmas
    :param freq: desired sampling frequency of output
    :param sheetname: optional parameter if data is not in first sheet
    :return: t: time vector of the signals
             yR: Right side plot
             yL: Left side plot
             lap: Information about start and end of each lap, matrix of size n x 2
             labels: label of each step in order
             firstC: first contact of each step, follows labels order
             lastC: last contact of each step, follows labels order
    """
    # names = ["num", "Index", "First Contact (sec.)", "Last Contact (sec.)", "Integ. Pressure (p x sec.)",
    #          "Foot Length (cm.)", "Foot Length %", "Foot Width (cm.)", "Foot Area (cm. x cm.)", "Foot Angle (degrees)",
    #          "Toe In/Out Angle (degrees)", "Foot Toe X Location (cm.)", "Foot Toe Y Location (cm.)",
    #          "Foot Heel X Location (cm.)", "Foot Heel Y Location (cm.)", "Foot Center X Location (cm.)",
    #          "Foot Center Y Location (cm.)", "Step Length (cm.)", "Absolute Step Length (cm.)", "Stride Length (cm.)",
    #          "Stride Width (cm.)", "Step Time (sec.)", "Stride Time (sec.)", "Stride Velocity (cm./sec.)",
    #          "DOP (degrees)", "Gait Cycle Time (sec.)", "Stance Time (sec.)", "Stance %", "Swing Time (sec.)",
    #          "Swing %", "Single Support (sec.)", "Single Support %", "Initial D. Support (sec.)",
    #          "Initial D. Support %", "Terminal D. Support (sec.)", "Terminal D. Support %", "Total D. Support (sec.)",
    #          "Total D. Support %", "CISP Time (sec.)", "CISP AP (%)", "CISP ML (%)", "Stance COP Dist. (cm.)",
    #          "SS COP Dist. (cm.)", "DS COP Dist. (cm.)", "Stance COP Dist. %", "SS COP Dist. %",
    #          "Stance COP Path Eff. %", "SS COP Path Eff. %", "DS COP Path Eff.%"]
    #
    # d = pandas.read_excel(f, sheetname=sheetname, header=None, names=names, skiprows=20, parse_cols=len(names) - 1)
    # print(d)
    if d is None:
        d = matFile2Pandas(f, sheetname)
    tEnd = d['Last Contact (sec.)'][d.shape[0] - 1] + 1
    t = np.linspace(0, tEnd, int(np.round((tEnd - 0) * freq)))
    yR = np.ones_like(t)
    yL = np.ones_like(t)
    # yR[0] = 0
    # yL[0] = 0
    # I don't think you'll do more than 100 laps
    lap = np.empty((100, 2))
    cL = 0
    print('shape of d: ', d.shape)
    for j in range(d.shape[0]):
        if np.isnan(d['First Contact (sec.)'][j]):
            continue
        lab = d['Index'][j]
        st = int(d['First Contact (sec.)'][j] * freq)
        en = int(d['Last Contact (sec.)'][j] * freq)
        if 'Right' in lab:
            yR[st:en + 1] = 0
        else:
            yL[st:en + 1] = 0
        if str(cL + 1) + ':' in lab:
            lap[cL, 0] = t[st]
            if cL > 0:
                lap[cL - 1, 1] = t[int(d['Last Contact (sec.)'][j - 1] * freq)]
            cL = cL + 1
    lap[cL - 1, 1] = t[en]
    lap = lap[:cL]
    labels = d['Index']
    firstC = d['First Contact (sec.)'].values
    lastC = d['Last Contact (sec.)'].values
    if usePercentage:
        yR = binary2Percentage(yR)
        yL = binary2Percentage(yL)
    return t, yR, yL, lap, labels, firstC, lastC


def createPlotsFromMat(d=None, f=None, freq=100, sheetname=0, usePercentage=False):
    """
    Creates signals representing foot on the groung or not on the ground
    :param d: Pandas Dataframe obtained from function
    :param f: File name of the excel file processed with pkmas
    :param freq: desired sampling frequency of output
    :param sheetname: optional parameter if data is not in first sheet
    :return: t: time vector of the signals
             yR: Right side plot
             yL: Left side plot
             lap: Information about start and end of each lap, matrix of size n x 2
             labels: label of each step in order
             firstC: first contact of each step, follows labels order
             lastC: last contact of each step, follows labels order
    """
    # names = ["num", "Index", "First Contact (sec.)", "Last Contact (sec.)", "Integ. Pressure (p x sec.)",
    #          "Foot Length (cm.)", "Foot Length %", "Foot Width (cm.)", "Foot Area (cm. x cm.)", "Foot Angle (degrees)",
    #          "Toe In/Out Angle (degrees)", "Foot Toe X Location (cm.)", "Foot Toe Y Location (cm.)",
    #          "Foot Heel X Location (cm.)", "Foot Heel Y Location (cm.)", "Foot Center X Location (cm.)",
    #          "Foot Center Y Location (cm.)", "Step Length (cm.)", "Absolute Step Length (cm.)", "Stride Length (cm.)",
    #          "Stride Width (cm.)", "Step Time (sec.)", "Stride Time (sec.)", "Stride Velocity (cm./sec.)",
    #          "DOP (degrees)", "Gait Cycle Time (sec.)", "Stance Time (sec.)", "Stance %", "Swing Time (sec.)",
    #          "Swing %", "Single Support (sec.)", "Single Support %", "Initial D. Support (sec.)",
    #          "Initial D. Support %", "Terminal D. Support (sec.)", "Terminal D. Support %", "Total D. Support (sec.)",
    #          "Total D. Support %", "CISP Time (sec.)", "CISP AP (%)", "CISP ML (%)", "Stance COP Dist. (cm.)",
    #          "SS COP Dist. (cm.)", "DS COP Dist. (cm.)", "Stance COP Dist. %", "SS COP Dist. %",
    #          "Stance COP Path Eff. %", "SS COP Path Eff. %", "DS COP Path Eff.%"]
    #
    # d = pandas.read_excel(f, sheetname=sheetname, header=None, names=names, skiprows=20, parse_cols=len(names) - 1)
    # print(d)
    try:
        if d is None:
            d = matFile2Pandas(f, sheetname)
        tEnd = d['Last Contact (sec.)'][d.shape[0] - 1] + 1
        t = np.linspace(0, tEnd, int(np.round((tEnd - 0) * freq)))
        yR = np.ones_like(t)
        yL = np.ones_like(t)
        # yR[0] = 0
        # yL[0] = 0
        # I don't think you'll do more than 100 laps
        lap = np.empty((100, 2))
        cL = 0
        for j in range(d.shape[0]):
            if np.isnan(d['First Contact (sec.)'][j]):
                continue
            lab = d['Index'][j]
            st = int(d['First Contact (sec.)'][j] * freq)
            en = int(d['Last Contact (sec.)'][j] * freq)
            if 'Right' in lab:
                yR[st:en + 1] = 0
            else:
                yL[st:en + 1] = 0
            if ':' in lab:
                lap[cL, 0] = t[st]
                if cL > 0:
                    lap[cL - 1, 1] = t[int(d['First Contact (sec.)'][j - 1] * freq)]
                cL = cL + 1
        lap[cL - 1, 1] = t[en]
        lap = lap[:cL]
        labels = d['Index']
        firstC = d['First Contact (sec.)'].values
        lastC = d['Last Contact (sec.)'].values
        if usePercentage:
            yR = binary2Percentage(yR)
            yL = binary2Percentage(yL)
    except:
        t, yR, yL, lap, labels, firstC, lastC = createPlotsFromMatForNew(d=d, f=f, freq=freq, sheetname=sheetname,
                                                                         usePercentage=usePercentage)
    return t, yR, yL, lap, labels, firstC, lastC


def time2ind(t, f, offset=0):
    """
    Return the index of time t sampled at frequency f, the function assumes that signals are sampled at constant freq
    :param t: Time
    :param f: Sample frequency
    :param offset: start of vector t, if time doesn't starts at 0
    :return: Index of time t
    """
    return np.rint((t - offset) * f).astype(int)


def randomOrder(n):
    shu = np.arange(n)
    np.random.shuffle(shu)
    return shu


def rotate1sampleStep(step):
    """
    Rotates one hole step from intertial frame to body frame, rotation is done inplace
    :param step: XYZ of all markers during one phase or step
    :return: rotated xyz of steps values
    """
    pEnd = step[-1, 3:6].copy()
    pStart = step[0, 3:6].copy()
    iHat = pEnd - pStart
    iHat[2] = 0
    rotMat = createRotMat(iHat, np.array([0.0, 0.0, 1.0]), normalVecs=True)
    for k in range(0, step.shape[1], 3):
        step[:, k:k + 3] = rotTrajectory(rotMat.T, step[:, k:k + 3])


def rotTrajectory(rotMat, tr):
    """
    Rotates a point trajectory using rotMat
    :param rotMat: Rotation matrix of size 3x3
    :param tr: trajectory of a point in time of size n x 3
    :return: rotated trajectory
    """
    trNew = np.dot(rotMat, tr.T).T
    # trNew = np.zeros_like(tr)
    # for j in range(tr.shape[0]):
    #     trNew[j, :] = np.dot(rotMat, tr[j, :].T)
    return trNew


def createRotMat(iVec, kVec, normalVecs=True):
    """
    Created a rotation matrix given i and k
    :param iVec: i Vector in Inertial frame
    :param kVec: k Vector in Inertial frame
    :param normalVecs: if True, both vectors are normalized
    :return: Rotation matrix
    """
    if normalVecs:
        iHat = iVec / np.linalg.norm(iVec)
        kHat = kVec / np.linalg.norm(kVec)
    else:
        iHat = iVec
        kHat = kVec
    jHat = np.cross(kHat, iHat) / np.linalg.norm(np.cross(kHat, iHat))
    rotMat = np.reshape(np.concatenate((iHat, jHat, kHat)), (3, 3)).T
    return rotMat


def reSampleMatrix(A, n, t=None, ti=None, axis=0):
    """
    Resample a complete matrix, this assumes that the signals are concatenated in a matrix along axis
    :param A: Matrix containg all signals
    :param n: desired length if time is not needed (ji.e. upsampling or downsampling)
    :param t: If t is not None Ti is required, orignal time vector
    :param ti: New time vector or desired time
    :param axis: axis where time is increasing
    :return: Resampled matrix Ai
    """
    if t is None:
        x = np.arange(A.shape[axis])
        xi = np.linspace(0, A.shape[axis] - 1, n)
    else:
        x = t
        xi = ti
    itp = interp1d(x, A, axis=axis)
    A2 = itp(xi)

    return A2


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, axis=0):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def randBetween(minVal, maxVal, shapeMat):
    """
    Return a random array of shape between min and max
    :param minVal: minimum possible value
    :param maxVal: maximum possible value
    :param shapeMat: tuple containing the shape to return
    :return: random array of shape between min and max
    """
    return minVal + np.random.random(shapeMat) * (maxVal - minVal)


def binary2Percentage(y, verbose=False):
    '''
    y is vector
    y is in logits, 0 is on the ground, 1 is not the ground
    '''
    # y(n+1)- y(n):
    # 0  --> no change
    # 1  --> toe off
    # -1 --> heel strike
    change = y[1:] - y[:-1]
    change[change > 0] = 1
    toeOff = np.where(change == 1)[0] + 1
    heelStrike = np.where(change == -1)[0] + 1
    # There is only one correct case: len(HS)> len(To) & Hs[0] < To[0]
    if len(heelStrike) == len(toeOff):
        if heelStrike[0] < toeOff[0]:
            toeOff = toeOff[:-1]
        else:
            toeOff = toeOff[1:]
    elif len(heelStrike) < len(toeOff):
        toeOff = toeOff[1:-1]
    if not (len(heelStrike) > len(toeOff) & heelStrike[0] < toeOff[0]):
        print('Something is wrong???!!!')
    yP = np.zeros_like(y)
    for i, (h, t) in enumerate(zip(heelStrike[:-1], heelStrike[1:])):
        yP[h:t] = np.linspace(0, 1, t - h)
    if verbose:
        tt = np.arange(y.size)
        f, axarr = plt.subplots(2, 1, sharex=True, sharey=True)
        axarr[0].plot(tt, y)
        axarr[1].plot(tt, yP)
        plt.show()
    # pResolution = 5
    # yP = np.round(yP * 100.0 / pResolution)
    return yP


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def removeNan(data):
    if np.sum(np.isnan(data)) == 0:
        return data
    print('Removing %d NaNs' % np.sum(np.isnan(data)))
    data[0, np.isnan(data[0, :])] = 0
    data[-1, np.isnan(data[-1, :])] = 0
    c = 0
    while np.any(np.isnan(data)):
        if c > 20:
            print('Failed to remove %d NaNs' % np.sum(np.isnan(data)))
            data[np.isnan(data)] = 0
            break
        c += 1
        for j in range(data.shape[1]):
            if np.any(np.isnan(data[:, j])):
                k = np.where(np.isnan(data[:, j]))[0]
                kk = k[1:] - k[:-1]
                if np.all(kk == 1):
                    data[k, j] = np.linspace(data[k[0] - 1, j], data[k[-1] + 1, j], k.size)
                    continue
                pk = np.where(kk > 1)[0]
                st = k[0]
                for ppk in pk:
                    en = k[ppk]
                    data[st:en + 1, j] = np.linspace(data[st - 1, j], data[en + 1, j], en - st + 1)
                    st = k[ppk + 1]
                en = k[-1]
                data[st:en + 1, j] = np.linspace(data[st - 1, j], data[en + 1, j], en - st + 1)
                #     if np.any(np.isnan(data[:, j])):
                #         for k in range(1, data.shape[0]-1):
                #             if np.isnan(data[k, j]):
                #                 data[k, j] = (data[k-1, j] + data[k+1, j]) / 2

    return data


def myDigitalFilter(y, t, desFreq):
    """
    Filters a digital signal to prevent changes faster than desFreq
    :param y: Signal
    :param t: time
    :param desFreq: cutoff frequency
    :return: filtered y signal with changes longer than 1 / desFreq
    """
    # print(t.shape)
    # print(y.shape)
    t = np.squeeze(t)
    newT = np.linspace(t[0], t[-1], num=int(np.round((t[-1] - t[0]) * desFreq)))
    yResam = np.round(np.interp(newT, t, y))
    newY = np.round(np.interp(t, newT, yResam))
    return newY


# def compareTemporalPandas(dRef, dShoe, **kwargs):
#     """
#     Compares 2 pandas dataset of temporal parameters
#     :param dRef: pandas Dataset with temporal parameters from reference system
#     :param dShoe: pandas Dataset with temporal parameters predicted by shoes
#     :return:
#     """
#     if 'verbose' in kwargs:
#         verbose = kwargs['verbose']
#     else:
#         verbose = False
#     if 'savePath' in kwargs:
#         savePath = kwargs['savePath']
#         if not os.path.exists(savePath):
#             os.makedirs(savePath)
#     else:
#         savePath = None
#     if 'name' in kwargs:
#         name = kwargs['name']
#     else:
#         name = 'Model'
#     columns = list(dRef)
#     if columns != list(dShoe):
#         print("Columns in dataset not equals!!! \n Cannot compare")
#     dataShoeCorr = pandas.DataFrame(columns=columns)
#     dataRefCorr = pandas.DataFrame(columns=columns)
#     dataDiff = pandas.DataFrame(columns=columns)
#     # compare by laps
#     for lap in range(1, int(min([dShoe['Lap_Index'].max(), dRef['Lap_Index'].max()])) + 1):
#         print(lap)
#         aRef = dRef.loc[dRef['Lap_Index'] == lap, :].copy()
#         aSho = dShoe.loc[dShoe['Lap_Index'] == lap, :].copy()
#         aRef.reset_index(inplace=True, drop=True)
#         aSho.reset_index(inplace=True, drop=True)
#         difSize = aRef.shape[0] - aSho.shape[0]
#         c = 0
#         skipLap = False
#         # remove extra steps
#         while difSize != 0:
#             try:
#                 if difSize < 0:  # Shoe has more steps, this is expected
#                     if aSho.iloc[0, 1] is not aRef.iloc[0, 1]:
#                         aSho.drop(aSho.index[0], inplace=True)
#                         difSize = aRef.shape[0] - aSho.shape[0]
#                     elif aSho.iloc[-1, 1] is not aRef.iloc[-1, 1]:
#                         aSho.drop(aSho.index[-1], inplace=True)
#                         difSize = aRef.shape[0] - aSho.shape[0]
#                     else:
#                         # find closest to start
#                         pos = (aSho.loc[aSho['isLeft'] == aRef.iloc[0, 1], 'HS'] - aRef.iloc[0, 3]).abs().idxmin()
#                         if pos != 0:
#                             aSho = aSho.iloc[1:, :]
#                             difSize = aRef.shape[0] - aSho.shape[0]
#                             continue
#                         # find closest to end
#                         pos = (aSho.loc[aSho['isLeft'] == aRef.iloc[-1, 1], 'HS'] - aRef.iloc[-1, 3]).abs().idxmin()
#                         if pos != 0:
#                             aSho = aSho.iloc[:-1, :]
#                             difSize = aRef.shape[0] - aSho.shape[0]
#                             continue
#                     aRef.reset_index(inplace=True, drop=True)
#                     aSho.reset_index(inplace=True, drop=True)
#                 if difSize > 0:  # Ref has more steps, this is not expected
#                     if aSho.iloc[0, 1] is not aRef.iloc[0, 1]:
#                         aRef.drop(aRef.index[0], inplace=True)
#                         difSize = aRef.shape[0] - aSho.shape[0]
#                     elif aSho.iloc[-1, 1] is not aRef.iloc[-1, 1]:
#                         aRef.drop(aRef.index[-1], inplace=True)
#                         difSize = aRef.shape[0] - aSho.shape[0]
#                     else:
#                         # find closest to start
#                         pos = (aRef.loc[aRef['isLeft'] == aSho.iloc[0, 1], 'HS'] - aSho.iloc[0, 3]).abs().idxmin()
#                         if pos != 0:
#                             aRef = aRef.iloc[1:, :]
#                             difSize = aRef.shape[0] - aSho.shape[0]
#                             continue
#                         # find closest to end
#                         pos = (aRef.loc[aRef['isLeft'] == aSho.iloc[-1, 1], 'HS'] - aSho.iloc[-1, 3]).abs().idxmin()
#                         if pos != 0:
#                             aRef = aRef.iloc[:-1, :]
#                             difSize = aRef.shape[0] - aSho.shape[0]
#                             continue
#                     aRef.reset_index(inplace=True, drop=True)
#                     aSho.reset_index(inplace=True, drop=True)
#                 c += 1
#                 if c > 30:
#                     skipLap = True
#                     break
#             except IndexError:
#                 skipLap = True
#                 break
#
#         if skipLap:
#             print('Error in lap %s' % lap)
#             continue
#
#         dataShoeCorr = pandas.concat((dataShoeCorr, aSho.copy()), ignore_index=True)
#         dataRefCorr = pandas.concat((dataRefCorr, aRef.copy()), ignore_index=True)
#         aDiff = aSho - aRef
#         aDiff.loc[:, ['Lap_Index', 'isLeft', 'Step_Index']] = aRef.loc[:, ['Lap_Index', 'isLeft', 'Step_Index']]
#         dataDiff = pandas.concat((dataDiff, aDiff.copy()), ignore_index=True)
#
#     if verbose or (savePath is not None):
#         for col in columns[3:-1]:
#             print(col)
#             try:
#                 plotT.plotErrorFigures(dataRefCorr[col].values, dataShoeCorr[col].values, removeOut=True,
#                                        savePlot=savePath, name=name + col)
#             except:
#                 print('Error in %s' % col)
#                 pass
#
#     return dataDiff, dataRefCorr, dataShoeCorr


def getAllTemporalParamsPandas(yRuncheck, yLuncheck, laps=None, minT=0.1, maxT=2.0, freq=100.0, subID=0, dt=0.0):
    '''
    y is vector
    y is in logits, 0 is on the ground, 1 is not the ground
    '''
    # y(n+1)- y(n):
    # 0  --> no change
    # 1  --> toe off
    # -1 --> heel strike
    # toeOff, heelStrike, swingDuration, stanceDuration, newY
    if laps is None:
        laps = np.reshape(np.array([0, yLuncheck.size / freq + 30]), (1, 2))
    parsR = getTemporalParams(yRuncheck, minT=minT, maxT=maxT, freq=freq)
    parsL = getTemporalParams(yLuncheck, minT=minT, maxT=maxT, freq=freq)
    # Single Support %,	Initial D. Support, Initial D. Support %, Terminal D. Support, Terminal D. Support %,
    # Total D. Support, Total D. Support %, Cadence (stp/min)
    cols = ['Lap_Index', 'isLeft', 'Step_Index', 'HS', 'TO', 'Step_time', 'Stride_time', 'Stance_Time',
            'Stance_Time%', 'Swing_Time', 'Swing_Time%', 'Single_Support', 'Single_Support%', 'Initial_D_Support',
            'Initial_D_Support%', 'Terminal_D_Support', 'Terminal_D_Support%', 'Total_D_Support',
            'Total_D_Support%', 'Cadence', 'SubID']
    # remove double steps
    colsMin = ['isLeft', 'HS', 'TO', 'drop']
    dataMin = pandas.DataFrame(data=np.full((parsR[1].shape[0] + parsL[1].shape[0], len(colsMin)), np.nan),
                               columns=colsMin)
    parsAll = [parsR, parsL]
    sideBoolAll = [False, True]
    c = 0
    for par, sideBool in zip(parsAll, sideBoolAll):
        lenSteps = par[1].shape[0]
        if sideBool:
            lBool = np.ones(lenSteps, dtype=np.bool)
        else:
            lBool = np.zeros(lenSteps, dtype=np.bool)
        dataMin.loc[c:c + lenSteps - 1, 'isLeft'] = lBool.copy()
        dataMin.loc[c:c + lenSteps - 1, 'HS'] = par[1].copy() / freq + dt
        dataMin.loc[c:c + lenSteps - 2, 'TO'] = par[0].copy() / freq + dt
        c += lenSteps
    dataMin['drop'] = False
    dataMin.dropna(inplace=True)
    dataMin.sort_values('HS', inplace=True)
    dataMin.reset_index(inplace=True, drop=True)
    for j in range(dataMin.shape[0] - 1):
        # if double step
        if dataMin.loc[j, 'isLeft'] is dataMin.loc[j + 1, 'isLeft']:
            dataMin.loc[j + 1, 'drop'] = True
            dataMin.loc[j, 'TO'] = dataMin.loc[j + 1, 'TO']
    dataMin = dataMin.loc[~dataMin['drop'], :]
    HSr = dataMin.loc[~dataMin['isLeft'], 'HS'].values
    TOr = dataMin.loc[~dataMin['isLeft'], 'TO'].values
    HSl = dataMin.loc[dataMin['isLeft'], 'HS'].values
    TOl = dataMin.loc[dataMin['isLeft'], 'TO'].values
    st = dataMin[['HS', 'TO']].min().min()
    en = dataMin[['HS', 'TO']].max().max()
    t = np.arange(st, en, 1.0 / freq)
    yR = createYfromHSTO(HSr, TOr, t)
    yL = createYfromHSTO(HSl, TOl, t)
    # redo without double steps
    parsR = getTemporalParams(yR, minT=minT, maxT=maxT, freq=freq)
    parsL = getTemporalParams(yL, minT=minT, maxT=maxT, freq=freq)
    data = pandas.DataFrame(data=np.full((parsR[1].shape[0] + parsL[1].shape[0], len(cols)), np.nan), columns=cols)
    SSR = np.logical_and(yR == 0, yL == 1).astype(np.int)
    SSL = np.logical_and(yL == 0, yR == 1).astype(np.int)
    DSB = np.logical_and(yR == 0, yL == 0).astype(np.int)
    # Add calculated parameters to dataframe
    parsAll = [parsR, parsL]
    sideBoolAll = [False, True]
    c = 0
    for par, sideBool, ss in zip(parsAll, sideBoolAll, [SSR, SSL]):
        lenSteps = par[1].shape[0]
        if sideBool:
            lBool = np.ones(lenSteps, dtype=np.bool)
        else:
            lBool = np.zeros(lenSteps, dtype=np.bool)
        stTime = (par[1][1:] - par[1][:-1]) / freq
        indAux = np.arange(lenSteps, dtype=np.int) + 1
        data.loc[c:c + lenSteps - 1, 'isLeft'] = lBool.copy()
        data.loc[c:c + lenSteps - 1, 'Step_Index'] = indAux.copy()
        data.loc[c:c + lenSteps - 1, 'HS'] = par[1].copy() / freq
        data.loc[c:c + lenSteps - 2, 'TO'] = par[0].copy() / freq
        data.loc[c:c + lenSteps - 2, 'Stride_time'] = stTime.copy()
        data.loc[c:c + lenSteps - 2, 'Stance_Time'] = par[3].copy()
        data.loc[c:c + lenSteps - 2, 'Swing_Time'] = par[2].copy()

        for j in range(par[1].size - 1):
            hsSt = par[1][j] - 1
            hsEn = par[1][j + 1]
            ssAux = ss[hsSt:hsEn]
            data.loc[c + j, 'Single_Support'] = np.sum(ssAux) / freq
            dsAux = DSB[hsSt:hsEn]
            change = dsAux[1:] - dsAux[:-1]
            dsUp = np.where(change == 1)[0] + 1
            dsDown = np.where(change == -1)[0] + 1
            if 0 < dsUp.shape[0] < 3 and 0 < dsDown.shape[0] < 3:
                data.loc[c + j, 'Initial_D_Support'] = np.sum(dsAux[:dsDown[0]]) / freq
                data.loc[c + j, 'Total_D_Support'] = np.sum(dsAux) / freq
                data.loc[c + j, 'Terminal_D_Support'] = data['Total_D_Support'].loc[c + j] - \
                                                        data['Initial_D_Support'].loc[c + j]
            elif dsUp.shape[0] == 0 and dsDown.shape[0] == 0:
                data.loc[c + j, 'Initial_D_Support'] = 0.0
                data.loc[c + j, 'Total_D_Support'] = 0.0
                data.loc[c + j, 'Terminal_D_Support'] = 0.0

                # sort Dataframe by time of HS
        c += lenSteps
    for col in cols:
        if '%' in col:
            data[col] = data[col[:-1]] / data['Stride_time'] * 100.0
    data.sort_values('HS', inplace=True)
    data.reset_index(inplace=True, drop=True)
    for j in range(data.shape[0] - 1):
        # calculate Step_Time if next step is not same side
        if data.loc[j, 'isLeft'] is not data.loc[j + 1, 'isLeft']:
            data.loc[j, 'Step_time'] = data['HS'][j + 1] - data['HS'][j]
            data.loc[j, 'Cadence'] = 60.0 / data['Step_time'][j]
    # cut into laps
    for j in range(laps.shape[0]):
        # do this per lap and double check...
        stLap = laps[j, 0] - 0.5
        enLap = laps[j, 1] + 0.5
        p = np.logical_and(data['HS'] >= stLap, data['TO'] <= enLap)
        data.loc[p, 'Lap_Index'] = j + 1
    data['SubID'] = subID
    return data


def getTemporalParams(y, minT=0.1, maxT=2.0, freq=100.0):
    '''
    y is vector
    y is in logits, 0 is on the ground, 1 is not the ground
    '''
    # y(n+1)- y(n):
    # 0  --> no change
    # 1  --> toe off
    # -1 --> heel strike
    change = y[1:] - y[:-1]
    change[change > 0] = 1
    toeOff = np.where(change == 1)[0] + 1
    heelStrike = np.where(change == -1)[0] + 1
    # There is only one correct case: len(HS)> len(To) & Hs[0] < To[0]
    if len(heelStrike) == len(toeOff):
        if heelStrike[0] < toeOff[0]:
            toeOff = toeOff[:-1]
        else:
            toeOff = toeOff[1:]
    elif len(heelStrike) < len(toeOff):
        toeOff = toeOff[1:-1]
    if not (len(heelStrike) > len(toeOff) & heelStrike[0] < toeOff[0]):
        print('Something is wrong???!!!')

    # remove steps that are too short
    # for j in range(10):
    #     swingDuration = heelStrike[1:] - toeOff
    #     if np.any(swingDuration < 0):
    #         print("Negative time found in swing!")
    #     s = 0.1 * j + 0.1
    #     inva = np.where(swingDuration <= minT * freq * s)[0]
    #     m = np.ones(len(swingDuration), dtype=bool)
    #     m[inva] = False
    #     toeOff = toeOff[m]
    #     m = np.ones(len(heelStrike), dtype=bool)
    #     m[inva + 1] = False
    #     heelStrike = heelStrike[m]
    stanceDuration = toeOff - heelStrike[:-1]
    if np.any(stanceDuration < minT * freq):
        for j in range(10):
            stanceDuration = toeOff - heelStrike[:-1]
            if np.any(stanceDuration < 0):
                print("Negative time found in stance!")
            s = 0.1 * j + 0.1
            # inva = np.where(np.logical_or(stanceDuration <= minT * freq * s, stanceDuration > maxT * freq))[0]
            inva = np.where(stanceDuration <= minT * freq * s)[0]
            m = np.ones(len(stanceDuration), dtype=bool)
            m[inva] = False
            toeOff = toeOff[m]
            m = np.ones(len(heelStrike), dtype=bool)
            m[inva] = False
            heelStrike = heelStrike[m]
    swingDuration = (heelStrike[1:] - toeOff)
    if np.any(swingDuration < minT * freq):
        for j in range(10):
            swingDuration = (heelStrike[1:] - toeOff)
            s = 0.1 * j + 0.1
            inva = np.where(swingDuration <= minT * freq * s)[0]
            m = np.ones(len(swingDuration), dtype=bool)
            m[inva] = False
            toeOff = toeOff[m]
            m = np.ones(len(heelStrike), dtype=bool)
            m[inva + 1] = False
            heelStrike = heelStrike[m]

    swingDuration = (heelStrike[1:] - toeOff) / freq
    stanceDuration = (toeOff - heelStrike[:-1]) / freq
    newY = np.zeros_like(y)
    for tO, hS in zip(toeOff, heelStrike[1:]):
        newY[tO:hS] = 1
    return toeOff, heelStrike, swingDuration, stanceDuration, newY


def createYfromHSTO(HeelStrikes, ToeOffs, t):
    """
    Creates a binary plot of feet on the ground or on the air
    :param HeelStrikes: Events for heel strikes
    :param ToeOffs: Events for Toe offs
    :param freq: Frequency for the output
    :return: time and function of time
    """
    pH = ~np.logical_or(np.isnan(HeelStrikes), np.isnan(ToeOffs))
    HeelStrikes = HeelStrikes[pH]
    ToeOffs = ToeOffs[pH]
    freq = 1.0 / (t[1] - t[0])
    y = np.ones_like(t)
    for tO, hS in zip(ToeOffs, HeelStrikes):
        tInd = int(np.round(tO * freq))
        hInd = int(np.round(hS * freq))
        y[hInd:tInd] = 0
    return y


def tTestTost(s1, s2, th):
    t = stats.ttest_ind(s1, s2)[1]
    to = smws.ttost_ind(s1, s2, -th, th, usevar='pooled')[0]
    # to = smws.ttost_paired(s1, s2, -th, th)[0]
    return t, to


def readViconCSV(fiName=None, st=2, syncCol=None, syncVal=4.0, cutAtSync=True, resampleFreq=None, filterFreq=None):
    """
    :param fiName: Path to the file. If None, a file_dialog is summoned
    :type fiName: str
    :param st: Value of names row
    :type st: int
    :param syncCol:  Name of the sync column. This column is assumed to be a float
    :type syncCol: str
    :param syncVal: Threshold value of the sync
    :type syncVal: float
    :param cutAtSync: Flag to remove values outside of the sync
    :type cutAtSync: bool
    :param resampleFreq: If is not None, signal is resampled to the desired freq
    :type resampleFreq: float
    :param filterFreq: If is not None, cutoff frequency for low pass filter
    :return: read data as a pandas DataFrame
    :rtype: pandas.DataFrame
    """
    if fiName is None:
        root = tk.Tk()
        #
        fiName = tk.filedialog.askopenfilename(parent=root, title='Select Violin binary file',
                                               filetypes=(("Binary files", "*.bin"), ("all files", "*.*")))
        root.withdraw()
        if fiName is None:
            return 0

    with open(fiName) as f:
        read_data = f.read()
    f.closed

    lines = read_data.split('\n')

    freq = float(lines[1])

    names = lines[2].split(',')

    names = [x for x in names if x != '']

    subNames = lines[3].split(',')[st:]

    units = lines[4].split(',')[st:]

    # assume all variable have x,y,z

    colName = []
    c = 0
    for n in names[:-1]:
        for i in range(3):
            sN = subNames[c]
            colName.append(n + ' ' + sN)
            c += 1

    # colName.append(subNames[c])

    data = np.array([np.fromstring(l, sep=',')[st:] for l in lines[5:-2]])
    if filterFreq is not None:
        data[:, :-1] = butter_lowpass_filter(data[:, :-1], filterFreq, freq)
    if resampleFreq is not None:
        tOld = np.arange(data.shape[0] / freq, step=1.0 / freq)
        tNew = np.arange(data.shape[0] / freq, step=1.0 / resampleFreq)
        data = reSampleMatrix(data, 0, t=tOld, ti=tNew)
        freq = resampleFreq
    dt = pandas.DataFrame(data=data, columns=colName)

    if syncCol is not None:
        dt[syncCol] = dt[syncCol] >= syncVal

        if cutAtSync:
            dt = dt.loc[dt[syncCol], :]
            dt.reset_index(inplace=True)

    return dt, units, freq


def compareYPhase(yT, tT, yP, tP, name='Plots', freq=100.0, laps=None, verbose=False, saveLaps=None, nLapsPlots=1):
    """
        Compares two synced on/off signals
        :param laps:
        :param yT: True values of y
        :param tT: Time of true values
        :param yP: Predicted values of y
        :param tP: Time of predicted signals
        :param name: Name string for this signals
        :param freq: sample frequency of the signals
        :param verbose: If true, it plot error comparison plots
        :return: Swing phase values for true and predicted steps
        """
    if laps is None:
        laps = np.reshape(np.array([tT[0], tT[-1]]), (1, 2))
    # delay = np.abs(np.correlate(yT, yP, "full")).argmax() - len(yP)
    # print(delay)
    # tP += delay*(tP[1] - tP[0])
    if verbose:
        f, axarr = plt.subplots(1, sharex=True, sharey=True)
        axarr.plot(tT, yT, label='Mat')
        axarr.plot(tP, yP, color='red', label='Shoes')
        axarr.legend()
        axarr.set_title('True values')
    # check that both y are at freq
    ts = [tT, tP]
    ys = [yT, yP]
    for i, (auxT, auxY) in enumerate(zip(ts, ys)):
        auxDt = auxT[1] - auxT[0]
        if auxDt != 1.0 / freq:
            newT = np.linspace(auxT[0], auxT[-1], num=int(np.round((auxT[-1] - auxT[0]) * freq)))
            yResam = np.round(np.interp(newT, auxT, auxY))
            ts[i] = newT.copy()
            ys[i] = yResam.copy()

    lapsInd = time2ind(laps, freq, offset=0)
    colors = ['green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    SWtAll = None
    SWpAll = None
    STtAll = None
    STpAll = None
    stSkip = 0
    stTot = 0
    yLapT = []
    yLapP = []
    for j in range(laps.shape[0]):
        # do this per lap and double check...
        stLap = lapsInd[j, 0]
        enLap = lapsInd[j, 1]
        yLapT.append(ys[0][stLap:enLap].copy())
        yLapP.append(ys[1][stLap:enLap].copy())
        if saveLaps is not None and j % nLapsPlots == 0:
            f, axarr = plt.subplots(2, sharex=True, sharey=True)
            axarr[0].plot(tT[stLap:enLap], yT[stLap:enLap])
            axarr[0].set_title('Lap %s Mat' % j)
            axarr[1].plot(tP[stLap:enLap], yP[stLap:enLap])
            axarr[1].set_title('Shoe')
            # for kk in range(len(HSt[:-1])):
            #     st1 = int(HSt[kk] + 1) + stLap
            #     en1 = int(HSt[kk + 1]) + stLap
            #     st2 = int(HSp[kk] + 1) + stLap
            #     en2 = int(HSp[kk + 1]) + stLap
            #     cInd = kk
            #     while cInd >= len(colors):
            #         cInd += -len(colors)
            #     axarr[0].plot(tT[st1:en1], yT[st1:en1], color=colors[cInd])
            #     axarr[1].plot(tP[st2:en2], yP[st2:en2], color=colors[cInd])
            plt.savefig(saveLaps + name + "lap%s" % j, dpi=300)
            # plt.show()
            # remove last

    yLapT = np.concatenate(yLapT, axis=0)
    yLapP = np.concatenate(yLapP, axis=0)
    yDiff = yLapT == yLapP
    accuracy = yDiff.mean()
    print(accuracy)


def createDirIfNotExist(filePath):
    if not os.path.exists(filePath):
        os.makedirs(filePath)


def readMatFileSeries(fiName=None):
    if fiName is None:
        root = tk.Tk()
        #
        fiName = tk.filedialog.askopenfilename(parent=root, title='Select Violin binary file',
                                               filetypes=(("Excel files", ("*.xls", "*.xlsx")), ("all files", "*.*")))
        root.withdraw()
        if fiName is None:
            return 0

    data = pandas.read_excel(fiName)
    inc2m = .0254**2
    for s in ['Right ', 'Left ']:
        data[s + 'Ground Reaction Force'] = data[s + 'Foot Pressure'] * data[s + 'Foot # Active Sensors'] * inc2m * 0.5**2

    return data


