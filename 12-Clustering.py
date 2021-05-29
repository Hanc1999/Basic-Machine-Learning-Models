from math import inf
import numpy as np
import time
from numpy import random

from scipy.stats import multivariate_normal
# from sklearn.metrics import normalized_mutual_info_score  # for validation


class Clustering():
    def __init__(self, k=3):
        self.k = k
        self.epi = 1e-10

    def load_data(self):
        temp = np.loadtxt("seeds_dataset.txt")
        x = temp[:, :temp.shape[1]-1]
        y = temp[:, temp.shape[1]-1]
        return x, y

    def cal_distance(self, arr1, arr2):
        return np.sqrt(np.sum(np.square(arr1-arr2)))

    def init_centroid(self, x):  # Generate random points for initialization
        # m = x.shape[0]
        n = x.shape[1]
        centroids = np.zeros((self.k, n))
        # Pick the first 3 rows
        centroids[:, :] = x[:self.k, :]

        # centroids[:, :] = x[m-self.k:m, :]

        # random.seed(10)
        # r = random.choice(range(m), self.k)
        # centroids[:, :] = x[r, :]

        # centroids[:, :] = np.random.randn(self.k, n)

        return centroids

    def kMeans(self):
        start = time.time()

        x, y = self.load_data()  # x: 210 * 7
        centroids = self.init_centroid(x)  # 3 * 7
        m = x.shape[0]  # The number of samples 210
        LabelDist = np.zeros((m, 2))  # Label - Distance array
        iteration = 0

        while True:
            iteration += 1
            onChange = False

            # Find the nearest center and calculate distance
            for i in range(m):
                sample = x[i, :]
                minDist = np.inf
                minLabel = -1
                for j in range(self.k):
                    center = centroids[j, :]
                    dist = self.cal_distance(sample, center)
                    if dist < minDist:
                        minDist = dist
                        minLabel = j + 1
                if (LabelDist[i, 0] != minLabel) or (LabelDist[i, 1] != minDist):
                    onChange = True
                    LabelDist[i, 0] = minLabel
                    LabelDist[i, 1] = minDist

            if not onChange:
                break

            # Update center
            label_all = LabelDist[:, 0]  # The label
            for i in range(self.k):
                idx = np.nonzero(label_all == (i + 1))
                x_label = x[idx]
                centroids[i, :] = np.mean(x_label, axis=0)  # calculate by columns

        print("K-means")
        print(f"Running time: {time.time() - start:.4f}")
        print(f"Iteration: {iteration}")
        print(f"Loss: {np.sum(LabelDist[:, 1]):.4f}")
        print(f"Purity: {self.purity(y, LabelDist[:, 0]):.4f}")
        print(f"Rand index: {self.rand_index(y, LabelDist[:, 0]):.4f}")
        print(f"Normalized mutual information: {self.normalized_mutual_information(y, LabelDist[:, 0]):.4f}\n")

    def KMeans_acce(self):
        start = time.time()

        x, y = self.load_data()
        centroids = self.init_centroid(x)
        m = x.shape[0]  # The number of samples 210
        n = x.shape[1]
        Label = np.zeros((m))  # Label array
        CenDist = np.zeros((self.k, self.k))  # 3 * 3

        LowerBound = np.zeros((m, self.k))  # 210 * 3
        UpperBound = np.zeros((m))  # 210 * 1
        RX = np.ones((m))  # 210 * 1

        # Assign each point to its cloest center
        for i in range(m):
            sample = x[i, :]
            minDist = np.inf
            minLabel = -1
            for j in range(self.k):
                center = centroids[j, :]
                dist = self.cal_distance(sample, center)
                LowerBound[i, j] = dist  # sample i to center j
                if dist < minDist:
                    minDist = dist
                    minLabel = j + 1
            UpperBound[i] = minDist
            Label[i] = minLabel

        iteration = 1

        # Repeat process until convergence
        while True:
            iteration += 1
            onChange = False
            SC = np.zeros((self.k))  # 3 * 1
            for i in range(self.k):
                for j in range(i, self.k):
                    if i == j:
                        CenDist[i, j] = inf
                    else:
                        CenDist[i, j] = self.cal_distance(centroids[i], centroids[j])
                        CenDist[j, i] = CenDist[i, j]
                SC[i] = np.min(CenDist[i])
            for i in range(self.k):
                CenDist[i, i] = 0
            SC /= 2

            xIdx = []
            for i in range(m):
                idx = int(Label[i])  # 123
                if UpperBound[i] > SC[idx-1]:
                    xIdx.append(i)

            for i in xIdx:
                x_sample = x[i, :]
                idx = int(Label[i])  # idx-1 ~ c(x)   bug ~ if i is from 1 - x_new all index are wrong!
                center = centroids[idx-1]  # c(x)
                for j in range(self.k):  # j ~ c

                    if (idx - 1 != j) and (UpperBound[i] > LowerBound[i, j]) and (UpperBound[i] > 0.5 * CenDist[idx-1, j]):  # bug
                        c = centroids[j]
                        if (RX[i]):
                            d = self.cal_distance(x_sample, center)  # d(x, c(x))
                            UpperBound[i] = d
                            RX[i] = 0
                        else:
                            d = UpperBound[i]
                        if (d > LowerBound[i, j]) or (d > 0.5 * CenDist[idx-1, j]):
                            dxc = self.cal_distance(x_sample, c)
                            LowerBound[i, j] = dxc
                            if dxc < d:
                                Label[i] = j + 1
                                UpperBound[i] = dxc

            # Update center
            MC = np.zeros((self.k, n))

            for i in range(self.k):
                idx = np.nonzero(Label == (i + 1))
                if np.shape(idx)[1] == 0:  # no related label
                    print(f"Label{i+1} is empty")
                    continue
                x_label = x[idx]
                MC[i, :] = np.mean(x_label, axis=0)  # calculate by columns

            newCenDist = np.zeros((self.k))  # center i to new center(MC) i
            for i in range(self.k):
                newCenDist[i] = self.cal_distance(centroids[i], MC[i])
                if newCenDist[i] > self.epi:
                    onChange = True

            for j in range(self.k):  # c
                c = centroids[j]
                cen_dist = newCenDist[j]
                for i in range(m):
                    LowerBound[i, j] = max(LowerBound[i, j] - cen_dist, 0)  # bug

            for i in range(m):
                idx = int(Label[i])
                cen_dist = newCenDist[idx-1]  # c(x)
                UpperBound[i] = UpperBound[i] + cen_dist
                RX[i] = 1
            centroids[:, :] = MC[:, :]

            if not onChange:
                break

        print("K-means - acceleration")
        print(f"Running time: {time.time() - start:.4f}")
        print(f"Iteration: {iteration}")
        print(f"Purity: {self.purity(y, Label):.4f}")
        print(f"Rand index: {self.rand_index(y, Label):.4f}")
        print(f"Normalized mutual information: {self.normalized_mutual_information(y, Label):.4f}\n")

    def soft_KMeans(self):
        start = time.time()

        x, y = self.load_data()  # x: 210 * 7
        centroids = self.init_centroid(x)  # 3 * 7
        m = x.shape[0]  # The number of samples 210
        n = x.shape[1]
        Degree = np.zeros((m, self.k))  # Degree of assignment, 210 * 3

        beta = 1
        iteration = 0

        while True:
            iteration += 1
            onChange = False

            # Assign degree based on responsibilities
            for i in range(m):
                sample = x[i, :]
                for j in range(self.k):
                    center = centroids[j, :]
                    dist = self.cal_distance(sample, center)
                    Degree[i, j] = np.exp(-beta * dist)
                dist_all = np.sum(Degree[i, :])
                Degree[i, :] /= dist_all

            # Update center
            mk = np.zeros((self.k, n))  # 3 * 7  x: 210*7  Degree:210*3
            for j in range(self.k):
                mk[j] = Degree[:, j].dot(x) / np.sum(Degree[:, j])

            # for i in range(m):
            #     sample = x[i, :]
            #     for j in range(self.k):
            #         mk[j, :] += Degree[i, j] * sample

            # for j in range(self.k):
            #     mk[j, :] /= np.sum(Degree[:, j])

            if self.cal_distance(mk, centroids) > self.epi:
                onChange = True

            centroids[:, :] = mk[:, :]

            if not onChange:
                break

        Label = Degree.argmax(axis=1) + 1
        print("Soft K-means")
        print(f"Iteration: {iteration}")
        print(f"Running time: {time.time() - start:.4f}")
        print(f"Purity: {self.purity(y, Label):.4f}")
        print(f"Rand index: {self.rand_index(y, Label):.4f}")
        print(f"Normalized mutual information: {self.normalized_mutual_information(y, Label):.4f}\n")

    def GMM_EM(self):
        print("Please wait......")
        start = time.time()

        x, y = self.load_data()  # x: 210 * 7
        m = x.shape[0]  # The number of samples 210
        n = x.shape[1]

        centroids = self.init_centroid(x)  # 3 * 7  m   miu
        weights = np.ones((m, self.k)) / self.k  # 210 * 3  y
        sig = np.zeros((self.k, n, n))  # 3 * 7 * 7   Σ
        mix = np.ones(self.k) / self.k  # Π

        for i in range(self.k):
            sig[i] = np.eye(n)

        iteration = 0
        loss = 0.0

        while True:
            iteration += 1

            # E step
            for i in range(m):
                weight = 0.0
                for j in range(self.k):
                    weights[i, j] = mix[j] * multivariate_normal.pdf(x[i, :], centroids[j, :], sig[j, :, :])
                    weight += weights[i, j]
                weights[i, :] /= weight

            # M step
            for j in range(self.k):
                # Update centroids
                weight = 0.0
                centroids_new = np.zeros(n)
                for i in range(m):
                    weight += weights[i, j]
                    centroids_new += weights[i, j] * x[i, :]
                centroids_new /= weight
                centroids[j, :] = centroids_new[:]

                # Update sigma
                sig_new = np.zeros((n, n))
                for i in range(m):
                    temp = np.matrix(x[i, :] - centroids[j, :])
                    sig_new += weights[i, j] * np.dot(np.transpose(temp), temp)
                sig_new /= weight
                sig[j, :, :] = sig_new[:, :]

                # Update Π
                mix[j] = weight / m

            # Loss
            temp_loss = 0
            for i in range(m):
                tmp = 0
                for j in range(self.k):
                    tmp += mix[j] * multivariate_normal.pdf(x[i, :], mean=centroids[j, :], cov=sig[j, :, :])
                temp_loss += np.log(tmp)

            if abs(temp_loss - loss) < self.epi:
                break

            loss = temp_loss
            # print(loss)
            
        Label = weights.argmax(axis=1) + 1
        print("Gaussian Mixture Model - Expectation Maximization")
        print(f"Iteration: {iteration}")
        print(f"Loss: {loss:.4f}")
        print(f"Running time: {time.time() - start:.4f}")
        print(f"Purity: {self.purity(y, Label):.4f}")
        print(f"Rand index: {self.rand_index(y, Label):.4f}")
        print(f"Normalized mutual information: {self.normalized_mutual_information(y, Label):.4f}\n")

    def purity(self, y, y_prec):
        n = len(y)
        confusion = np.zeros((self.k, self.k))
        for i in range(n):
            idx = int(y[i])
            idx_prec = int(y_prec[i])
            confusion[idx-1, idx_prec-1] += 1
        # print(confusion)
        pure = np.sum(np.max(confusion, axis=1)) / n
        return pure

    def rand_index(self, y, y_prec):
        n = len(y)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(n):
            for j in range(i+1, n):
                if (y[i] == y[j]) and (y_prec[i] == y_prec[j]):
                    TP += 1
                elif (y[i] != y[j]) and (y_prec[i] != y_prec[j]):
                    TN += 1
                elif (y[i] != y[j]) and (y_prec[i] == y_prec[j]):
                    FP += 1
                elif (y[i] == y[j]) and (y_prec[i] != y_prec[j]):
                    FN += 1
        RI = (TP + TN) / (TP + TN + FP + FN)
        return RI

    def normalized_mutual_information(self, y, y_prec):
        n = len(y)

        HY = np.zeros((self.k))
        HY_prec = np.zeros((self.k))
        count = np.zeros((self.k))
        confusion = np.zeros((self.k, self.k))

        for i in range(n):
            idx = int(y[i])
            idx_prec = int(y_prec[i])
            confusion[idx-1, idx_prec-1] += 1

        HY = np.sum(confusion, axis=1)
        HY_prec = np.sum(confusion, axis=0)
        count = np.sum(confusion, axis=1)

        for i in range(len(HY)):
            if HY[i] == 0:
                continue
            HY[i] = HY[i] / n
            HY[i] = -HY[i] * np.log2(HY[i])
        for i in range(len(HY_prec)):
            if HY_prec[i] == 0:
                continue
            HY_prec[i] = HY_prec[i] / n
            HY_prec[i] = -HY_prec[i] * np.log2(HY_prec[i])
        hy = np.sum(HY)
        hy_prec = np.sum(HY_prec)

        HY_cond = np.zeros((self.k))

        for i in range(self.k):
            for j in range(self.k):
                if confusion[i, j] == 0:
                    continue
                confusion[i, j] = confusion[i, j] / count[i]
                confusion[i, j] = -confusion[i, j] * np.log2(confusion[i, j])

        for i in range(self.k):
            HY_cond[i] = count[i] / n * np.sum(confusion[i, :])
        hy_cond = np.sum(HY_cond)
        IY = hy - hy_cond
        NMI = 2 * IY / (hy + hy_prec)
        # print("Check NMI:", normalized_mutual_info_score(y, y_prec))
        return NMI


def main():
    C = Clustering()
    C.kMeans()
    C.KMeans_acce()
    C.soft_KMeans()
    C.GMM_EM()


if __name__ == '__main__':
    main()
