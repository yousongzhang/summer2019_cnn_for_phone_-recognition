import mfcc_convert_timit
import glob
import os
import timit



path = timit.TIMIT_PATH + "/TRAIN/"
print("path", path)
pattern = os.path.join(path, "*/*/*.WAV")
files = glob.glob(pattern)
# Standard practic is to remove all "sa" sentences
# for each speaker since they are the same for all.


for f in files:
    print("processing train ", f)
    mfcc_convert_timit.create_mfcc(f)



path = timit.TIMIT_PATH + "/TEST/"
print("path", path)
pattern = os.path.join(path, "*/*/*.WAV")
files = glob.glob(pattern)
# Standard practic is to remove all "sa" sentences
# for each speaker since they are the same for all.


for f in files:
    print("processing test", f)
    mfcc_convert_timit.create_mfcc(f)
