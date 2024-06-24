

def rgb2grayscale(data):
    # V = 0.30*R + 0.59G + 0.11*B
    return 0.30 * data[:, :, 0] + 0.59 * data[:, :, 1] + 0.11 * data[:, :, 2]
