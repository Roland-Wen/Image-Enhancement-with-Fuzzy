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

def SSFCF(image, N, SC):
    assert N % 2 == 1, "N must be an odd number"
    
    def muMore(z, alpha, beta):
        assert 0.0 <= z <= 1.0, f"z = {z} which is out of bound for muMore"
        return 1 / (1 + np.exp(-alpha * z + beta))
    
    def getLambdas(block, fuzzySets, numOfRule = 9):
        assert block.shape == (N, N), "wrong shape to get lambdas"
        
        middle = int(block[N//2, N//2])
        lambdas = [0] * numOfRule
        
        # lambdas for NB~PB
        for rule in range(1, 7):
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
            lambdas[rule] = medianMu * muMore(cnt / (N * N - 1), alpha1, beta1)
        
        # lambdas for R'1 and R'2
        for rule in range(7, 9):
            numZbigger, numAbigger, minMuA = 0, 0, 1000.0
            for i in range(N):
                for j in range(N):
                    if (i, j) == (N//2, N//2): continue
                
                    xi = block[i][j] - middle
                    muA = fuzzySets[rule](xi)
                    muZ = fuzzySets[0](xi)
                    
                    if muA > muZ:
                        minMuA = min(minMuA, muA)
                        numAbigger += 1
                    elif muZ > muA:
                        numZbigger += 1
                        
            if minMuA > 2.0: minMuA = 0
            lambdas[rule] = minMuA * muMore(numZbigger / (N * N - 1), alpha1, beta1) * muMore(numAbigger / (N * N), alpha3, beta3)
        
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
    alpha3, beta3 = 15, 3
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
    diff.A1 =          R(               0,              256)
    diff.A2 =          S(            -256,                0)
    diff.B1 =          S(            -256, base + delta * 1)
    diff.B2 =          R(base + delta * 7,              256)
    
    diffFuzzySet = [diff.Z, diff.NB, diff.NM, diff.NS, 
                    diff.PS, diff.PM, diff.PB,
                    diff.A1, diff.A2, diff.B1, diff.B2]
    center = [0, base + delta * 1, base + delta * 2, base + delta * 3, 
              base + delta * 5, base + delta * 6, base + delta * 7, 256,
              -256, -256, 256]
    
    # plt.figure()
    # for fuzzy in diffFuzzySet:
    #     fuzzy.plot()
    
    ans = image.copy()
    # calculate lambdas and y
    for i in range(1, H-1):
        for j in range(1, W-1):
            currBlock = image[i-N//2:i+N//2+1, j-N//2:j+N//2+1]
            lambdas = getLambdas(currBlock, diffFuzzySet)
            currAvg, k = avg(currBlock, diff.Z)
            k = muMore(k / (N * N - 1), alpha2, beta2)
            y = k * currAvg
            y += (1 - k) * (
                     sum(lambdas[l] * center[l]     for l in range(1, 7)) +
                SC * sum(lambdas[l] * center[l + 2] for l in range(7, 9)))
            ans[i, j] += int(y)
            ans[i, j] = max(0, min(ans[i, j], 255))
            # if not 0 <= ans[i, j] <= 256:
            #     print(f"resulting pixel out of bound:ans[{i}][{j}]={ans[i][j]}")
            #     print(f"y={y}")
            #     print(f"lambdas={lambdas}")
            #     print(f"k={k}")
            #     print((1-k)*SC * lambdas[7] * center[7 + 2])
            #     assert False
                
    return ans
    
if __name__ == '__main__':
    image = cv2.imread('lenaNoise.png', 0)
    # image = np.array([[0, 128, 0 ,0],
    #                   [0, 128, 0, 0],
    #                   [0, 128, 0, 0],
    #                   [0, 128, 0 ,0]])
    plt.figure()
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 256)
    
    result = SSFCF(image, 3, 1)
    
    plt.figure()
    plt.imshow(result, cmap = 'gray', vmin = 0, vmax = 256)
    cv2.imwrite('SSFCF_lenaNoise.png', result)