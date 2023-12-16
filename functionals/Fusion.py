import numpy as np
from MachineLearning import kneighbors
import time

def ssm_fusion(ssm_1, ssm_2, nssm_1, nssm_2, k, t):
    print("\n********** Local SSM fusion start ***********")
    length = ssm_1.shape[0]

    skm_1 = np.zeros((length, length), dtype='float64')  # km means kernel matrix
    skm_2 = np.zeros((length, length), dtype='float64')

    f1_neighbors = kneighbors(ssm_1, length, k)
    f2_neighbors = kneighbors(ssm_2, length, k)

    print("sparse kernel matrix construction start...")
    # 1st feature based sparse kernel matrix construction
    for i in range(length):
        f1_ith_neighs = f1_neighbors[i]
        skm_1[i][f1_ith_neighs] = ssm_1[i][f1_ith_neighs] / np.sum(ssm_1[i][f1_ith_neighs])

        f2_ith_neighs = f2_neighbors[i]
        skm_2[i][f2_ith_neighs] = ssm_2[i][f2_ith_neighs] / np.sum(ssm_2[i][f2_ith_neighs])
    print("1st feature based skm has been completed")
    print("2nd feature based skm has been completed\n")

    print("fused ssm construction start...")
    # make normalized weight matrices by iterating t times

    st = time.time()

    for _t in range(t):
        print("time step : ", _t)
        temp = nssm_1.copy()
        nssm_1 = np.matmul(np.matmul(skm_1, nssm_2.copy()), skm_1.T)
        nssm_2 = np.matmul(np.matmul(skm_2, temp), skm_2.T)

    fused_ssm = (nssm_1 + nssm_2) / 2

    print(f"{time.time() - st:.4f} sec")  # 종료와 함께 수행시간 출력
    print("Done")
    print("**********************************************")
    return fused_ssm

def concatenate_fusion(f1, f2):
    return np.concatenate((f1,f2), axis = 1)