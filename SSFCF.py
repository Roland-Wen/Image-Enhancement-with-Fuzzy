import cv2
import matplotlib.pyplot as plt
import numpy as np
from fuzzylogic.classes import Domain
from fuzzylogic.functions import R, S, triangular
from statistics import median
# authors:
# - family-names: "Kiefner"
#   given-names: "Anselm"
#   orcid: "https://orcid.org/0000-0002-8009-0733"
# title: "FuzzyLogic for Python"
# version: 1.2.0
# doi: 10.5281/zenodo.6881817
# date-released: 2022-02-15
# url: "https://github.com/amogorkon/fuzzylogic"

def SFCF(image, N):
    def muMore(z, alpha, beta):
        assert 0.0 <= z <= 1.0, f"z = {z} which is out of bound for muMore"
        return 1 / (1 + np.exp(-alpha * z + beta))
    
    def getLambdas(block, fuzzySets):
        assert block.shape == (N, N), "wrong shape to get lambdas"
        
        middle = int(block[N//2, N//2])
        numOfRule = len(fuzzySets)
        lambdas = [0] * numOfRule
        
        for rule in range(1, numOfRule):
            members = []
            for i in range(N):
                for j in range(N):
                    if (i, j) == (N//2, N//2): continue
                    curr = fuzzySets[rule](block[i][j] - middle)
                    if curr > 0.0: members.append(curr)
            cnt = len(members)
            if cnt > 0:
                medianMu = median(members)
            else:
                medianMu = 0
            lambdas[rule] = medianMu * muMore(cnt / (N * N), alpha1, beta1)
        
        return lambdas
    
    # get the average and number of xi that belongs to rule
    def avg(block, rule):
        assert block.shape == (N, N), "wrong shape to avg"
        ans, cnt = 0.0, 0
        middle = int(block[N//2, N//2])
        for i in range(N):
            for j in range(N):
                if (i, j) == (N//2, N//2): continue
                curr = rule(block[i][j] - middle)
                if curr > 0.0:
                    ans += (block[i][j] - middle)
                    cnt += 1
        return ans / max(cnt, 1), cnt
    
    base, delta = -256, 512//8
    alpha1, beta1 = 13, 7
    alpha2, beta2 = 7, 3
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
            lambdas = getLambdas(image[i-1:i+2, j-1:j+2], diffFuzzySet)
            currAvg, k = avg(image[i-1:i+2, j-1:j+2], diff.Z)
            k = muMore(k / (N * N), alpha2, beta2)
            y = k * currAvg
            y += (1 - k) * sum(
                lambdas[l] * center[l] for l in range(len(center)))
            ans[i, j] += int(y)
            assert 0 <= ans[i,j] <= 256, f"resulting pixel out of bound:ans[{i}][{j}]={ans[i][j]}"

    return ans
    
if __name__ == '__main__':
    image = cv2.imread('lenaNoise.png', 0)
    # image = np.array([[0, 0, 0],
    #                   [0, 30, 0],
    #                   [0, 0, 0]])
    # row, col = image.shape
    # gauss = np.random.normal(0,20,(row,col))
    # gauss = gauss.reshape(row,col)
    # image += np.int32(gauss)
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 256)
    
    result = SFCF(image, 3)
    
    plt.figure()
    plt.imshow(result, cmap = 'gray', vmin = 0, vmax = 256)
    cv2.imwrite('SFCF_lenaSNP.png', result)