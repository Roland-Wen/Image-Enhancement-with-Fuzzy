import cv2
import matplotlib.pyplot as plt
import numpy as np
from fuzzylogic.classes import Domain
from fuzzylogic.functions import R, S, triangular
# authors:
# - family-names: "Kiefner"
#   given-names: "Anselm"
#   orcid: "https://orcid.org/0000-0002-8009-0733"
# title: "FuzzyLogic for Python"
# version: 1.2.0
# doi: 10.5281/zenodo.6881817
# date-released: 2022-02-15
# url: "https://github.com/amogorkon/fuzzylogic"

def FCF(image, N):
    def muMore(z, alpha, beta):
        assert 0.0 <= z <= 1.0, f"z = {z} which is out of bound for muMore"
        return 1 / (1 + np.exp(-alpha * z + beta))
    
    def getLambdas(block, fuzzySets):
        assert block.shape == (N, N), "wrong shape to get lambdas"
        
        middle = int(block[N//2, N//2])
        numOfRule = len(fuzzySets)
        lambdas = [None] * numOfRule
        
        for rule in range(1, numOfRule):
            minMu, cnt = 10000.0, 0
            for i in range(N):
                for j in range(N):
                    if (i, j) == (N//2, N//2): continue
                    curr = fuzzySets[rule](block[i][j] - middle)
                    if curr > 0.0:
                        minMu = min(minMu, curr)
                        cnt += 1
            if minMu > 1.1: minMu = 0
            lambdas[rule] = minMu * muMore(cnt / (N * N), alpha1, beta1)
        
        lambdas[0] = max(0, 1 - sum(lambdas[1:]))
        
        return lambdas
    
    base, delta = -256, 512//8
    alpha1, beta1 = 13, 7
    # alpha2, beta2 = 7, 3
    H, W = image.shape
    
    # construct membership functions
    diff = Domain('pixel difference', -256, 256, res = 0.1)
    diff.Z  = triangular(base + delta * 3, base + delta * 5)
    diff.NB =          S(base + delta * 1, base + delta * 2)
    diff.NM = triangular(base + delta * 1, base + delta * 3)
    diff.NS = triangular(base + delta * 2, base + delta * 4)
    diff.PS = triangular(base + delta * 4, base + delta * 6)
    diff.PM = triangular(base + delta * 5, base + delta * 7)
    diff.PB =          R(base + delta * 6, base + delta * 7)
    diffFuzzySet = [diff.Z, diff.NB, diff.NM, diff.NS, 
                    diff.PS, diff.PM, diff.PB]
    center = [0, base + delta * 1, base + delta * 2, base + delta * 3, 
              base + delta * 5, base + delta * 6, base + delta * 7]
        
    # for fuzzy in diffFuzzySet:
    #     fuzzy.plot()
    
    ans = image.copy()
    # calculate lambdas and y
    for i in range(1, H-1):
        for j in range(1, W-1):
            lambdas = getLambdas(image[i-1: i+2, j-1: j+2], diffFuzzySet)
            y = int(sum(lambdas[k] * center[k] for k in range(len(center))))
            ans[i, j] += y
            assert 0 <= image[i,j] <= 256, "resulting pixel out of bound"

    return ans
    
if __name__ == '__main__':
    image = cv2.imread('lenaNoise.png', 0)
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 256)
    
    result = FCF(image, 3)
    
    plt.figure()
    plt.imshow(result, cmap = 'gray', vmin = 0, vmax = 256)
    