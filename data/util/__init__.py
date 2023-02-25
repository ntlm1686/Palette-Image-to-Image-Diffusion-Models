import cv2
import sys
import numpy as np


def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def imread(path, chn='rgb', dtype='float32'):
    '''
    Read image.
    chn: 'rgb', 'bgr' or 'gray'
    out:
        im: h x w x c, numpy tensor
    '''
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    if chn.lower() == 'rgb':
        if im.ndim == 3:
            im = bgr2rgb(im)
        else:
            im = np.stack((im, im, im), axis=2)
    elif chn.lower() == 'gray':
        assert im.ndim == 2

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype ==  'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    if im.shape[2] > 3:
        im = im[:, :, :3]

    return im


def readline_txt(txt_file):
    if txt_file is None:
        out = []
    else:
        with open(txt_file, 'r') as ff:
            out = [x[:-1] for x in ff.readlines()]
    return out
