#!/usr/bin/env python
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import datetime

import cv2
import imutils
import simpleaudio as sa
from sklearn.cluster import AgglomerativeClustering

from piano import read_ref_audio

FPS = 20
MIN_AREA = 1000 #500
saveFrameDelay = 10 #2*FPS?


class SidewalkKeys(object):
    """This follows the motion tracking example provided here:
    https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
    """
    def __init__(self,refpath,prefix,
                 device=0,
                 notes=['C4','D4','E4','F4','G4'],
                 notelength=2.0,
                 samplefreq=44100,
                ):
        self.device = device
        self.refpath = refpath
        self.prefix = prefix
        self.notes = notes
        self.notelength = notelength
        self.samplefreq = samplefreq
        # camera object
        self.camera = None
        # current camera frame
        self.frame = None
        # running average to get background
        self.avgframe = None
        # load raw sound data for each note
        self._load_rawdata()

    def _load_rawdata(self):
        self.rawdata = {
            note: read_ref_audio(
                note,
                refpath=self.refpath,
                prefix=self.prefix,
                sampleperiod=self.notelength,
            )
            for note in self.notes
        }

    def setup(self,delay=0,reverseorder=True):
        """Take a snapshot to use for identifying keyboard layout

        If reverseorder, then keys will go from left to right looking
        _at_ the camera instead of looking from the camera.
        """
        # capture one frame
        self.camera = cv2.VideoCapture(self.device)
        for _ in range(delay*FPS//1000):
            grabbed,frame = self.camera.read()
            if not grabbed:
                print('Problem getting camera frame')
                break
            key = cv2.waitKey(1000//FPS) & 0xFF
        region = cv2.selectROI("Select keyboard region and press any key...",
                               frame,showCrosshair=False)
        cv2.waitKey(0)
        self.stop()
        # determine key extents
        print(region)
        x0,y0,width,height = region
        key_bounds = [int(fval) for fval in np.linspace(x0,x0+width,len(self.notes)+1)]
        self.keywidth = key_bounds[1] - key_bounds[0]
        if reverseorder:
            notes = self.notes[::-1]
        else:
            notes = self.notes
        self.keys = {
            note: Key(
                name=note,
                sample=self.rawdata[note], 
                xlim=(key_bounds[i], key_bounds[i+1]),
                ylim=(y0, y0+height),
                samplefreq=self.samplefreq,
            )
            for i,note in enumerate(notes)
        }
        # plot resulting keys
        fig,ax = plt.subplots(figsize=(12,8))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #rect = Rectangle((x0,y0), width, height, edgecolor=[0,0,1], fill=None)
        #ax.add_patch(rect)
        def plot_key(name):
            keyx0 = self.keys[name].xlim[0]
            rect = Rectangle((keyx0,y0), self.keywidth, height, edgecolor=[0,0,1], fill=None)
            ax.add_patch(rect)
            xtext = int(keyx0 + self.keywidth/2)
            ytext = int(y0 + height/2)
            ax.text(xtext, ytext, name, fontsize=16, color=[0,0,1],
                    horizontalalignment='center',
                    verticalalignment='center')
        for name in self.notes:
            plot_key(name)

    def start(self,
              show_feed=True,
              show_background=False,
              show_framedelta=False,
              show_threshold=False,
             ):
        """Start camera and video identification"""
        self.camera = cv2.VideoCapture(self.device)
        if not self.camera.isOpened():
            self.camera.open(self.device)
        while True:
            grabbed,newframe = self.camera.read()
            self.frame = newframe
            if not grabbed:
                print('Problem getting camera frame')
                break
            # process new frame
            bkg = self._extract_background(newframe)
            ref = self._blur_grayscale(bkg)
            gray = self._blur_grayscale(newframe)
            thresh,delta,regions = self._threshold(ref, gray)
            pts = self._identify_activity(regions)
            for key in self.keys.values():
                key.play_if_hit(pts)
            # update video panels
            if show_feed:
                cv2.imshow('Video feed', newframe)
            if show_background:
                cv2.imshow('Background', bkg)
            if show_framedelta:
                cv2.imshow('Abs delta', delta)
            if show_threshold:
                cv2.imshow('Threshold', thresh)
            # if the esc key is pressed, break from the loop
            key = cv2.waitKey(1000//FPS) & 0xFF
            if key == 27: #escape
                break
        self.stop()

    def stop(self):
        """Clean up environment"""
        cv2.destroyAllWindows()
        cv2.waitKey(1) # https://answers.opencv.org/question/102328/destroywindow-and-destroyallwindows-not-working/
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.avgframe = None

    def _extract_background(self,frame,alpha=0.5):
        """
        alpha: weight of input image: higher values mean that newer frames are
            weighted more, i.e., the background can change more quickly

        e.g., http://opencvpython.blogspot.com/2012/07/background-extraction-using-running.html
        """
        if self.avgframe is None:
            self.avgframe = np.float32(frame)
        else:
            cv2.accumulateWeighted(frame, self.avgframe, alpha)
        return cv2.convertScaleAbs(self.avgframe)

    def _blur_grayscale(self,frame0,resize=None,ksize=21):
        """
        resize: None or image width (int)
        ksize: Gaussian kernel size, assumed equal in both directions
        """
        # resize (optional)
        if resize is not None:
            assert isinstance(resize,int)
            frame = imutils.resize(frame0, width=resize)
        else:
            frame = frame0
        # convert from blue-green-red to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur, with automatically computed standard deviations
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        return blurred

    def _threshold(self,frame0,frame,cutoff=25,dilations=2):
        """
        Most of the heavy lifting happens here
        """
        # compute absolute difference between frames
        delta = cv2.absdiff(frame0, frame)
        # calculate threshold, maxval==255
        retval,thresh = cv2.threshold(delta, cutoff, 255, cv2.THRESH_BINARY)
        # dilate the thresholded image to fill holes
        # - kernel==None ==> simple 3x3 matrix
        if dilations > 0:
            thresh = cv2.dilate(thresh, None, iterations=dilations)
        # get contours
        # - mode==RETR_EXTERNAL: extreme outer contours only
        # - method==CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and
        #   diagonal segments and leaves only their end points. For example, an
        #   up-right rectangular contour is encoded with 4 points.
        contours, _ = cv2.findContours(thresh.copy(),
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        # loop over contours
        regions = []
        for cnt in contours:
            # if contour is too small--ignore it
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            else:
                regions.append(cnt)
            # compute bounding box
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        return thresh,delta,regions
        
    def _identify_activity(self,regions):
        ptslist = []
        for cnt in regions:
            pts = cnt.squeeze()
            x = np.mean(pts[:,0])
            y = np.mean(pts[:,1])
            ptslist.append(np.array([x,y]))
            cv2.putText(self.frame, 'x', (int(x),int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                        thickness=3)
        if len(ptslist) > 2:
            pts = np.array(ptslist)
            clustering = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=2*self.keywidth).fit(pts)
            Nclusters = len(np.unique(clustering.labels_))
            #print(np.unique(clustering.labels_))
            newpts = [np.zeros(2) for _ in range(Nclusters)]
            counts = np.zeros(Nclusters)
            for ic,pt in zip(clustering.labels_, ptslist):
                newpts[ic] += pt
                counts[ic] += 1
            newpts = [(newpts[ic] / counts[ic]).astype(int) for ic in range(Nclusters)]
        elif len(ptslist) == 1:
            pt = ptslist[0]
            newpts = [[int(pt[0]), int(pt[1])]]
        else:
            newpts = []
        for ic,pt in enumerate(newpts):
            cv2.putText(self.frame, str(ic), (pt[0],pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0),
                        thickness=3)
        return newpts


class Key(object):
    def __init__(self, name, sample, xlim, ylim,
                 samplefreq=44100,
                ):
        self.name = name
        self.sample = sample
        self.xlim = xlim
        self.ylim = ylim
        self.samplefreq = samplefreq
        self.buffer = None

    def play_if_hit(self,pts):
        for pt in pts:
            if (pt[0] > self.xlim[0]) and (pt[0] < self.xlim[1]) and \
               (pt[1] > self.ylim[0]) and (pt[1] < self.ylim[1]):
                self.play()

    def play(self):
        if self.buffer is not None:
            if self.buffer.is_playing():
                self.buffer.stop()
        self.buffer = sa.play_buffer(self.sample, 1, 2, self.samplefreq)


