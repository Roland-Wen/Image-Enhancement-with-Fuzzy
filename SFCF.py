import sys
from tqdm import tqdm
import cv2
import numpy as np
from os.path import exists
from fuzzylogic.classes import Domain
from fuzzylogic.functions import R, S, triangular
from statistics import median


def SFCF(image: np.ndarray, N=3) -> np.ndarray:
    """Enhance the given image using SFCF.

    If the argument `N` isn't passed in, the default is 3.

    Parameters
    ----------
    image : 2D numpy array
        The image that needs to be enhanced
    N : Odd intger
        The size of the mask for SFCF
    """

    assert N % 2 == 1, "N must be an odd integer"

    # Constants, 1 is for more, 2 is for more'
    base, delta = -256, 512//8
    alpha1, beta1 = 13, 7
    alpha2, beta2 = 7, 3
    H, W = image.shape

    def muMore(z: float, alpha: float, beta: float) -> float:
        """Calculate the function that has the form 1/(1+e^(-alpha*z+beta)).

        Parameters
        ----------
        z : float
            The input to be calculated
        alpha : float
            The steepness parameter of the function
        beta : float
            The shifting parameter of the function
        """

        return 1 / (1 + np.exp(-alpha * z + beta))

    def getLambdas(block: np.ndarray, fuzzySets: list) -> list:
        """Calculate firing level for each rule for this block.

        Parameters
        ----------
        block : 2D numpy array
            The block to be calculated
        fuzzySets : list of membership functions
            Membeship functions that would be used to calculate the firing
            levels
        """

        middle = int(block[N//2, N//2])
        numOfRule = len(fuzzySets)
        lambdas = [0] * numOfRule

        # Calculate lambda for each rule
        for rule in range(1, numOfRule):
            # Fuzzy value that is greater than 0 for this rule
            members = []
            for i in range(N):
                for j in range(N):
                    # Don't include myself
                    if (i, j) == (N//2, N//2):
                        continue

                    # Get the fuzzy value for this xi
                    curr = fuzzySets[rule](block[i][j] - middle)
                    if curr > 0.0:
                        members.append(curr)

            # Find median and size of member
            cnt = len(members)
            if cnt > 0:
                medianMu = median(members)
            else:
                medianMu = 0

            # Calculate lambda for this rule
            lambdas[rule] = medianMu * muMore(cnt/(N*N), alpha1, beta1)

        return lambdas

    # get the average and number of xi that belongs to rule
    def avg(block: np.ndarray, rule) -> (float, int):
        """Calculate the average of xi such that the membership of xi in rule
        is greater than 0.
        Also returns the number of xi such that the membership of xi in rule
        is greater than 0.

        Parameters
        ----------
        block : 2D numpy array
            The block to be calculated
        ruke : membership function
            Membeship functions that would be used to calculate the membership
        """

        total, cnt = 0.0, 0
        middle = int(block[N//2, N//2])

        # Loop through neighboor to get avg
        for i in range(N):
            for j in range(N):
                # Don't include myself
                if (i, j) == (N//2, N//2):
                    continue

                # Get the fuzzy value for this xi and update total
                curr = rule(block[i][j] - middle)
                if curr > 0.0:
                    total += (block[i][j] - middle)
                    cnt += 1

        return total / max(cnt, 1), cnt

    # construct membership functions
    diff = Domain("pixel difference", -256, 256, res=0.1)
    diff.Z = triangular(base + delta * 3, base + delta * 5)
    diff.NB = S(base + delta * 1, base + delta * 2)
    diff.NM = triangular(base + delta * 1, base + delta * 3)
    diff.NS = triangular(base + delta * 2, base + delta * 4)
    diff.PS = triangular(base + delta * 4, base + delta * 6)
    diff.PM = triangular(base + delta * 5, base + delta * 7)
    diff.PB = R(base + delta * 6, base + delta * 7)
    diffFuzzySet = [diff.Z, diff.NB, diff.NM, diff.NS,
                    diff.PS, diff.PM, diff.PB]
    center = [0, base + delta * 1, base + delta * 2, base + delta * 3,
              base + delta * 5, base + delta * 6, base + delta * 7]

    ans = image.copy()
    # Loop through all pixels and update them
    for i in tqdm(range(1, H-1)):
        for j in range(1, W-1):
            # Extract the current block that we're dealing with
            currBlock = image[i-N//2:i+N//2+1, j-N//2:j+N//2+1]
            # Get lambdas for R1~R6
            lambdas = getLambdas(currBlock, diffFuzzySet)
            # Deal with R0'
            currAvg, k = avg(currBlock, diff.Z)
            k = muMore(k / (N * N), alpha2, beta2)
            # Defuzzify
            y = k * currAvg
            y += (1 - k) * sum(
                lambdas[L] * center[L] for L in range(len(center)))
            ans[i, j] += int(y)

    return ans


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        print("Wrong number of arguments! Expected 1 or 2, but got " +
              f"{len(sys.argv)}")
        sys.exit()

    fileName = sys.argv[1]
    if not exists(fileName):
        print(f"{fileName} does not exist!")
        sys.exit()

    if len(sys.argv) == 3:
        if not sys.argv[2].isdigit():
            print(f"{sys.argv[2]} is not a positive integer!")
            sys.exit()

        N = int(sys.argv[2])
        if N % 2 == 0:
            print("N must be an odd integer!")
            sys.exit()
    else:
        N = 3

    image = cv2.imread(fileName, 0)
    result = SFCF(image, N)
    cv2.imwrite(f"SFCF_{fileName}", result)
    print(f"Created SFCF_{fileName}")
