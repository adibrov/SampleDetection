import numpy as np

def normalizePercentile(data, hi=99.9, lo=0.01, saturate=True, eps=1e-8):
    hiLim = np.percentile(data, hi)
    loLim = np.percentile(data, lo)
    nom = -.1
    denom = 1

    res = (data - loLim)/(hiLim - loLim + eps)

    if (saturate):
    	res = np.minimum(np.maximum(res,0),1)
    return hiLim, loLim, res