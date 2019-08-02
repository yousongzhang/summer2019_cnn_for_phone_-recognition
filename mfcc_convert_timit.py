from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt

import librosa
import librosa.display
import sys
import pandas as pd



def write_to_file(header,  mfcc, f):
    s = ""

    for i in mfcc:
        s += str(i) + " "
    s.rstrip()
    f.write(header + s + "\n")

def load_phone_map():
    with open("phones.60-48-39.map", 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [l for l in lines if len(l) == 3]
    m60_48 = {l[0] : l[1] for l in lines}
    m48_39 = {l[1] : l[2] for l in lines}
    return m60_48, m48_39



path = ""
input_phn = ""
output = ""
output_phn = ""


# if len(sys.argv) > 1 :
#     path = sys.argv[1]
# else:
#     print("missing input audio file path")
#     sys.exit()

def create_mfcc(path):
    paths = path.split('.', 2)

    if len(paths) == 2:
        output = paths[0] + ".mfcc"
        input_phn = paths[0] + ".PHN"
        output_phn = paths[0] + ".mfcc"
    else:
        print("missing input audio file path")
        sys.exit()

    print("processing audio file: ", path)

    y, sr = librosa.load(path,  sr=16000)

    # Set the hop length; at 22050 Hz, 220 samples ~= 10ms
    hop_length = int(sr/100)


    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length = hop_length)


    #np.savetxt(output, mfccs, delimiter=",")

    #print("save mfcc file: ", output)


    df=pd.read_csv(input_phn, sep=' ',header=None)

    data = df.values

    last = data[-1,1]
    rate = 16000
    row_sample = rate/100

    row = int(last/row_sample)


    current_row = 0
    p_start = data[current_row, 0]*100/row_sample
    p_end = data[current_row, 1]*100/row_sample


    f = open(output_phn, "w")
    m60_48, _ = load_phone_map()


    for x in range(0, row):
        start = x*100
        end = (x+1)*100

        f_start = start * 1.6
        f_end =  end * 1.6

        if end > p_end :
            current_row += 1
            p_start = data[current_row, 0]*100/row_sample
            p_end = data[current_row, 1]*100/row_sample
            header = "" + str(x*100) + " " + str((x+1)*100) +   " - "
            write_to_file(header, mfccs[:,x], f)
        else:
            phonemes = data[current_row, 2]
            phonemes = m60_48[phonemes]
            header = "" + str(x*100) + " " + str((x+1)*100) + " " +  phonemes + " "
            write_to_file(header, mfccs[:,x], f)

    f.close()
    print("saved to ", output_phn)
