from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import random






def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.PHN")
    files = glob.glob(pattern)
    # Standard practic is to remove all "sa" sentences
    # for each speaker since they are the same for all.
    filt_sa = lambda x : os.path.basename(x)[:2] != "sa"
    files = filter(filt_sa, files)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip() for l in fid)
            phonemes = (l.split()[-1] for l in lines)
            phonemes = [m60_48[p] for p in phonemes if p in m60_48]
            data[f] = phonemes
    return data


if __name__ == "__main__":

    #train = load_transcripts(os.path.join(path, "train"))
    path = "/Users/zhangyousong/Downloads/data/lisa/data/timit/raw/TIMIT/"
    load_transcripts(os.path.join(path, "TRAIN"))

