#!/usr/bin/env python
#
# Functions for playing tones from piano keys
#
import os
import struct
import aifc

import numpy as np
import simpleaudio as sa


def read_ref_audio(note, refpath='.', prefix='', sampleperiod=1.0):
    """Expect aiff inputs to be stereo, 2 bytes per signal for now"""
    fpath = os.path.join(refpath,'{:s}.{:s}.aiff'.format(prefix,note))

    # read ref note
    with aifc.open(fpath,'rb') as f:
        # nchannels: 1 or 2 (mono/stereo)
        # sampwidth: bytes per sample
        # framerate: sampling freq [frames/s]
        print(f.getparams())
        assert (f.getnchannels() == 2)  # stereo
        assert (f.getsampwidth() == 2)  # signal is represented by signed short values
        fs = f.getframerate()
        Nframes = f.getnframes()
        print('sample length:',Nframes/fs,'s')
        N = int(sampleperiod*f.getframerate())

        channel0 = []
        channel1 = []
        for i in range(N):
            frame = f.readframes(1)
            s0, s1 = struct.unpack('>hh',frame)
            channel0.append(s0)
            channel1.append(s1)
    channel0 = np.array(channel0)
    channel1 = np.array(channel1)

    # check FFT here (TODO), convert to mono
    Nfft = int(N/2)
    dt = 1/fs
    freq = np.fft.fftfreq(N, d=dt)
    F0 = np.fft.fft(channel0)
    F1 = np.fft.fft(channel1)
    Fmean = (F0 + F1) / 2
    u = np.fft.ifft(Fmean).astype(np.int16)

    return u


