#!/usr/bin/env python
import sys,os
import numpy as np
import datetime

import cv2
import imutils
import simpleaudio as sa

from piano import read_ref_audio

FPS = 20
MIN_AREA = 500 # default: 500
MAX_AREA_PCT = 0.9
alpha = 0.5

#saveFrameDelay = 10 #2*FPS
saveFrameDelay = 5


class SidewalkKeys(object):
    def __init__(self,refpath,prefix,
                 notes=['C4','D4','E4','F4','G4'],
                 notelength=2.0,
                 samplefreq=44100.,
                ):
        self.refpath = refpath
        self.prefix = prefix
        self.notes = notes
        self.notelength = notelength
        self.samplefreq = samplefreq
        self._load_rawdata()

    def _load_rawdata(self):
        self.rawdata = {
            name: read_ref_audio(
                name,
                refpath=self.refpath,
                prefix=self.prefix,
                sampleperiod=self.notelength,
            )
            for name in self.notes
        }

    def start(self,dev=0):
        camera = cv2.VideoCapture(dev)
        if not camera.isOpened():
            camera.open(dev)
        while True:
            (grabbed,frame0) = camera.read()
            cv2.imshow("Video feed", frame0)
            # if the esc key is pressed, break from the loop
            key = cv2.waitKey(1000//FPS) & 0xFF
            if key == 27: #escape
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1) # https://answers.opencv.org/question/102328/destroywindow-and-destroyallwindows-not-working/
        camera.release()

