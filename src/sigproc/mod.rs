// def preemphasis(signal, coeff=0.95):
//     return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

use ndarray::prelude::*;

pub fn preemphasis(signal: ndarray::Array1<f32>, coeff: f32) -> ndarray::Array1<f32> {
//     perform preemphasis on the input signal.
//     :param signal: The signal to filter.
//     :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
//     :returns: the filtered signal.
    let new_vector: Array1<_> = signal;
    let mut slice = new_vector.slice_mut(s![..1, ..]).to_owned();
    slice = slice - (coeff * slice);
    new_vector
}

pub fn rolling_window(a: ndarray::Array1<f32>, window: usize, step: u32) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
    let r = Array::from_elem((a.shape()[0], window), 0.);
    for index in window - 1 .. a.shape()[0] {
        let row = r.row_mut(index);
        row = a.slice_mut(s![(index-window+1)..index+1]);
    }
    r
}
//     # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
//     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
//     strides = a.strides + (a.strides[-1],)
//     return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

pub fn framesig(signal: ndarray::Array1<f32>, frame_len: f32, frame_step: f32, winfunc: Box::<dyn Fn(u32) -> ndarray::Array1<f32>>) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
//     Frame a signal into overlapping frames.
//     :param signal: the audio signal to frame.
//     :param frame_len: length of each frame measured in samples.
//     :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
//     :param winfunc: the analysis window to apply to each frame. By default no window is applied.
//     :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
//     :returns: an array of frames. Size is NUMFRAMES by frame_len.
    let signal_length = signal.len() as u32;
    let frame_len = frame_len.round() as u32;
    let frame_step = frame_step.round() as u32;
    let numframes = if signal_length <= frame_len { 1 } else { 1 + (signal_length - frame_len) / frame_step } as u32;
    let pad_length = (numframes - 1) * frame_step + frame_len;

    let zeros = Array::zeros((pad_length - signal_length) as usize);
    let pad_signal = stack![Axis(1), signal, zeros];
    let win = winfunc(frame_len);
    let frames = rolling_window(pad_signal, frame_len as usize, frame_step);
    frames * win
}

pub fn powspec(frames: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>, ndarray::Dim<[usize; 2]>>, NFFT: u32) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
    // def powspec(frames, NFFT):
    // """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    // :param frames: the array of frames. Each row is a frame.
    // :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    // :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    // """
    1.0 / NFFT as f32 * magspec(frames, NFFT).mapv(|a| a.powi(2))
}

pub fn magspec(frames: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>, NFFT: u32) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
    // def magspec(frames, NFFT):
    // """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    // :param frames: the array of frames. Each row is a frame.
    // :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    // :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    // """
    // if numpy.shape(frames)[1] > NFFT:
    //     logging.warn(
    //         'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
    //         numpy.shape(frames)[1], NFFT)
    // complex_spec = numpy.fft.rfft(frames, NFFT)
    // return numpy.absolute(complex_spec)
}