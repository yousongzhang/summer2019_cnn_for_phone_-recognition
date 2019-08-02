import mfcc_convert_timit
import glob
import os


# path = "/Users/zhangyousong/Downloads/data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV"
# mfcc_convert_timit.create_mfcc(path)
# path = "/Users/zhangyousong/Downloads/data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA2.WAV"
# mfcc_convert_timit.create_mfcc(path)


path = "/Users/zhangyousong/Downloads/data/lisa/data/timit/raw/TIMIT/TEST/"
pattern = os.path.join(path, "*/*/*.WAV")
files = glob.glob(pattern)
# Standard practic is to remove all "sa" sentences
# for each speaker since they are the same for all.


for f in files:
    mfcc_convert_timit.create_mfcc(f)