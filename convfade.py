#!/usr/local/bin/python3
import sys
import argparse
from os import path
import numpy as np
import librosa

def main():
    """ Main routine. """

    # --------------------------------------------------------------------------
    # Argument parsing and error checking:
    # --------------------------------------------------------------------------

    arg_parser = argparse.ArgumentParser(description="ConvFade")
    arg_parser.add_argument("-s", "--start", type=str, default="", \
            help="File path to audio clip that starts the fade.")
    arg_parser.add_argument("-e", "--end", type=str, default="", \
            help="File path to audio clip that ends the fade.")
    arg_parser.add_argument("-o", "--output", type=str, default="", \
            help="Output audio file path.")
    arg_parser.add_argument("-l", "--length", type=float, default=3.0, \
            help="Length of crossfade desired in seconds (default: 3.0).")
    arg_parser.add_argument("-f", "--frame", type=int, default=100, \
            help="Length of FFT frames desired in milliseconds (default: 100).")
    args = arg_parser.parse_args()

    if args.start == "" or not path.exists(args.start):
        sys.exit("Invalid start audio path: \"" + args.start + "\"")
    if args.end == "" or not path.exists(args.end):
        sys.exit("Invalid end audio path: \"" + args.end + "\"")
    if args.output == "":
        sys.exit("No output path given.")

    start_w, start_sr = librosa.core.load(args.start, sr=None, mono=True)
    end_w, end_sr = librosa.core.load(args.end, sr=None, mono=True)

    start_dur = start_w.shape[0] / float(start_sr)
    end_dur = end_w.shape[0] / float(end_sr)

    if args.length > start_dur or args.length > end_dur:
        sys.exit("Length of fade greater than length of at least " \
            + "one of the input audio clips.")
    if args.frame > args.length * 1000:
        sys.exit("Length of FFT frames is greater than the length " \
            + "of the crossfade.")

    # --------------------------------------------------------------------------
    # Match sample rates:
    # --------------------------------------------------------------------------

    # Global sample rate conforms to highest sample rate
    sr = start_sr
    if start_sr > end_sr:
        end_w = librosa.core.resample(end_w, end_sr, start_sr)
        sr = start_sr
    if end_sr > start_sr:
        start_w = librosa.core.resample(start_w, start_sr, end_sr)
        sr = end_sr

    # --------------------------------------------------------------------------
    # Accomplish fade:
    # --------------------------------------------------------------------------

    # Fade length in number of samples
    fade_len = int(sr * args.length)

    # Sample index in start audio at which fade begins
    fade_start = (start_w.shape[0] - 1) - fade_len

    # Frame length in number of samples
    frame_len = int((args.frame / 1000.) * sr)

    start_stft = librosa.core.stft(start_w[fade_start:], n_fft=frame_len).T
    end_stft = librosa.core.stft(end_w[0:fade_len], n_fft=frame_len).T
    assert start_stft.shape == end_stft.shape, "STFT shapes not equal."

    # Calculate ConvFade
    half_way = start_stft.shape[0] / 2
    ones_array = np.ones_like(start_stft[0])
    lowerpasser = np.arange(0, 3)
    result_stft = []
    for i, _ in enumerate(start_stft):
        if i < half_way:
            start_scale = 1.
            end_scale = float(i) / half_way
        else:
            start_scale = float(half_way - (i-half_way)) / half_way
            end_scale = 1.
        start_filtered = start_stft[i]
        end_filtered = end_stft[i]
        start_filtered[lowerpasser] = 0
        end_filtered[lowerpasser] = 0
        convolved = np.multiply(start_scale * start_stft[i] + ones_array, \
                end_scale * end_stft[i] + ones_array)
        result_stft.append(convolved)

    result_stft = np.array(result_stft)
    result_fade = librosa.core.istft(result_stft.T)

    # --------------------------------------------------------------------------
    # Export result:
    # --------------------------------------------------------------------------

    result = np.append(start_w[0:fade_start], result_fade)
    result = np.append(result, end_w[fade_len - 1:])

    librosa.output.write_wav(args.output, result, sr, norm=True)

if __name__ == "__main__":
    main()
