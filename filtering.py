import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  

from scipy.signal import butter,filtfilt
from scipy import signal
import matplotlib.pyplot as plt


''' given a signal, apply a bandpass filter and return the transformed signal''' 
def apply_bandpass(low, high, sig, samplerate, typ):
    # Filter requirements.
    fs = samplerate     # sample rate, Hz
    T = len(sig)/fs       # Sample Period
    
    # provide your filter frequencies in Hz
    cutoff = [low, high]

    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2  # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples
    def butter_bandpass_filter(data, cutoff, fs, order, typ = 'bandpass'):
        # get the filter coefficients 
        if(typ == 'bandpass'):
            b, a = butter(order, [cutoff[0]/nyq, cutoff[1]/nyq], btype=typ, analog=False)
        else:
            b, a = butter(order, [cutoff[0]/nyq], btype='lowpass', analog=False)
        y = filtfilt(b, a, data)
        return y
    y = butter_bandpass_filter(sig, cutoff, fs, order, typ)
    return y

''' remove systematic noise from a signal '''
def get_detrend(sig):
    detrend = signal.detrend(sig)
    return detrend

''' Low pass filter with a specified freq Wn '''
def remove_noise(data, f):
    N  = 2  
    B, A = signal.butter(N, f, output='ba')
    smooth_data = signal.filtfilt(B,A, data)
    return smooth_data

''' find peaks in a given signal and plot (optional) ''' 
def peak_finder(cleaned, plot = False):
    peaks = signal.find_peaks(cleaned, height = 0.7*max(cleaned))
    
    if(plot == True):
        plt.figure(figsize = (20,10))
        plt.scatter(peaks[0], cleaned[peaks[0]], color = 'r', label = 'peaks')
        plt.plot(cleaned, label = 'filtered, detrended Y signal')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Processed Y signal')
        plt.grid()
        plt.show()
    return peaks

''' Get the envelope of a signal using the naive peaks'''
def envelope_extractor(signal, plot = False):
    peaks =  peak_finder(signal)[0]
    peaks_n =  peak_finder(-signal)[0]
    signal = (signal - signal.min())/(signal.max() - signal.min())

    if(plot==True):
        plt.figure(figsize = (20,10))
        plt.title('Envelope Representation')
        plt.plot(peaks,signal[peaks], color = 'r')
        plt.plot(peaks_n,signal[peaks_n], color = 'g')

        plt.plot(signal)
        plt.grid()
        plt.show()
        
    return peaks, peaks_n, signal

''' Break signal into components of length new_len with a stride of step '''
def segment(arr, axis, new_len, step=1):

    old_shape = np.array(arr.shape)

    assert new_len <= old_shape[axis],  \
        "new_len is bigger than input array in axis"
    seg_shape = old_shape.copy()
    seg_shape[axis] = new_len

    steps = np.ones_like(old_shape)
    if step:
        step = np.array(step, ndmin = 1)
        assert step > 0, "Only positive steps allowed"
        steps[axis] = step

    arr_strides = np.array(arr.strides)

    shape = tuple((old_shape - seg_shape) // steps + 1) + tuple(seg_shape)
    strides = tuple(arr_strides * steps) + tuple(arr_strides)

    arr_seg = np.squeeze(
        as_strided(arr, shape = shape, strides = strides))

    # squeeze will move the segmented axis to the first position
    arr_seg = np.moveaxis(arr_seg, 0, axis)


    return arr_seg.copy()

''' Find reverse mapping from tau to theta'''
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

''' function to break audio signal into frames with a step size '''
from numpy.lib import stride_tricks
def segment(arr, axis, new_len, step=1):

    old_shape = np.array(arr.shape)

    assert new_len <= old_shape[axis],  \
        "new_len is bigger than input array in axis"
    seg_shape = old_shape.copy()
    seg_shape[axis] = new_len

    steps = np.ones_like(old_shape)
    if step:
        step = np.array(step, ndmin = 1)
        assert step > 0, "Only positive steps allowed"
        steps[axis] = step

    arr_strides = np.array(arr.strides)

    shape = tuple((old_shape - seg_shape) // steps + 1) + tuple(seg_shape)
    strides = tuple(arr_strides * steps) + tuple(arr_strides)

    arr_seg = np.squeeze(
        stride_tricks.as_strided(arr, shape = shape, strides = strides))

    # squeeze will move the segmented axis to the first position
    arr_seg = np.moveaxis(arr_seg, 0, axis)


    return arr_seg.copy()


    return arr_seg.copy()

''' generate Hilbert Transform ''' 
def get_hilbert(sig):
    analytic_signal = signal.hilbert((sig))
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi) * samplerate)
    
    return analytic_signal, amplitude_envelope, instantaneous_phase, instantaneous_frequency

''' get frequencies and amplitudes from the FFT'''
def get_f_Y(amp):
    n = len(amp) # length of the signal
    k = np.arange(n)
    Fs = 250e3
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[:len(frq)//2] # one side frequency range

    Y = np.fft.fft(amp)/n # dft and normalization
    Y = Y[:n//2]
    return frq, Y

''' get energy in a specific frequency band given FFT arrays '''
def get_breathing_band(f,Y, samplerate):
    min_speech = 35e3
    max_speech = 40e3

    start_speech =  find_nearest(f, min_speech)
    end_speech =  find_nearest(f, max_speech)

    e1 = np.abs(Y)[start_speech:end_speech]

    ratio = (np.sum(np.absolute(e1)**2))
    return ratio

''' normalize a signal to 0-1 '''
def normalize(signal):
    return  (signal-min(signal))/(max(signal)-min(signal))

def gcc_methods(sig, refsig, fs=1, max_tau=None, interp=16, beamform = False):
    '''
    Generalized Cross Correlation: regular CC, PHAT, SCOT and ROTH processors
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0]+ refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig)#, n=n)
    REFSIG = np.fft.rfft(refsig)#, n=n)
    R = SIG * np.conj(REFSIG)

    
    # standard gcc
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    
    if(beamform == True):
        cc = np.fft.irfft(R, n=(interp * n))
    if(beamform == 'SCOT'):
        cc = np.fft.irfft(R/np.sqrt((SIG*np.conj(SIG)*REFSIG*np.conj(REFSIG))), n=(interp * n))
    if(beamform == 'ROTH'):
        cc = np.fft.irfft(R/((SIG*np.conj(REFSIG))), n=(interp * n))
        

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift #/ float(interp * fs)
    
    return tau, cc

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

