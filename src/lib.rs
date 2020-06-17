#[macro_use]
extern crate ndarray;
pub mod sigproc;

fn calculate_nfft(samplerate: u32, winlen: u32) -> u32 {
    // Calculates the FFT size as a power of two greater than or equal to
    // the number of samples in a single window length.
    
    // Having an FFT less than the window length loses precision by dropping
    // many of the samples; a longer FFT than the window allows zero-padding
    // of the FFT buffer which is neutral in terms of frequency domain conversion.
    // :param samplerate: The sample rate of the signal we are working with, in Hz.
    // :param winlen: The length of the analysis window in seconds.
    let window_length_samples = winlen * samplerate;
    let mut nfft: u32 = 1;
    while nfft < window_length_samples {
        nfft *= 2;
    }
    nfft
}

fn hz2mel(hz: f32) -> f32 {
    // Convert a value in Hertz to Mels
    // :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    // :returns: a value in Mels. If an array was passed in, an identical sized array is returned.

    2595.0 * (1.0+hz/700.0).log10()
}

fn mel2hz(mel: f32) -> f32 {
    // Convert a value in Mels to Hertz
    // :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    // :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    700.0*(10_f32.powf(mel/2595.0)-1.0)
}

// def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
//     nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
//     winfunc=lambda x:numpy.ones((x,))):
// """Compute MFCC features from an audio signal.
// :param signal: the audio signal from which to compute features. Should be an N*1 array
// :param samplerate: the sample rate of the signal we are working with, in Hz.
// :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
// :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
// :param numcep: the number of cepstrum to return, default 13
// :param nfilt: the number of filters in the filterbank, default 26.
// :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
// :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
// :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
// :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
// :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
// :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
// :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
// :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
// """
fn mfcc<S>(signal: ndarray::Array1<f32>, samplerate: u32, winlen: f32, winstep: f32, numcep: u32, nfilt: u32, nfft: u32, lowfreq: u32, highfreq: u32, preemph: f32, ceplifter: u32, appendEnergy: bool, winfunc: u32) {
    let (feat, energy) = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc);
    // feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    // if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    // return feat
}

// def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
//     nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
//     winfunc=lambda x:numpy.ones((x,))):
// """Compute Mel-filterbank energy features from an audio signal.
// :param signal: the audio signal from which to compute features. Should be an N*1 array
// :param samplerate: the sample rate of the signal we are working with, in Hz.
// :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
// :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
// :param nfilt: the number of filters in the filterbank, default 26.
// :param nfft: the FFT size. Default is 512.
// :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
// :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
// :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
// :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
// :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
//   second return value is the energy in each frame (total energy, unwindowed)
// """
fn fbank(signal: ndarray::Array1<f32>, samplerate: u32, winlen: f32, winstep: f32, nfilt: u32, nfft: u32, lowfreq: u32, highfreq: u32, preemph: f32, winfunc: u32) -> (ndarray::Array1<S>, u32) {
    signal = sigproc::preemphasis(signal,preemph);
    let frames = sigproc::framesig(signal, winlen*(samplerate as f32), winstep*(samplerate as f32), winfunc);
    let pspec = sigproc::powspec(frames,nfft);
    energy = numpy.sum(pspec,1); // this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) ;// if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq);
    feat = numpy.dot(pspec,fb.T) // compute the filterbank energies;
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat); // if feat is zero, we get problems with log

    feat,energy
}