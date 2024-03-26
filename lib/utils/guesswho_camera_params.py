import numpy as np


def get_guesswho_K():
    # Ref: https://github.com/anastasia-tkach/honline-cpp-public/blob/5bd4fe19ecfcfc299aebc0741850eadd95833dcf/tracker/Data/Camera.cpp#L18
    K = np.array(
        [[287.26, 0, 320/2],
        [0, 287.26, 240/2],
        [0, 0, 1]]
    , dtype=np.float32)
    return K