# coding=UTF-8
import numpy as np
import sys
import time


def spreadAoD(mu, std):
    b=std/np.sqrt(2)
    a=np.random.rand()-0.5
    x=mu-b*np.sign(a)*np.log(1-2*abs(a))
    return x


def prc(t, beta):
    if abs(t) == 1 / (2 * beta):
        p = np.pi / 4 * np.sinc(1 / (2 * beta))
    else:
        p = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
    return p


def MU_MISO_OFDM_MODEL(Ncl,Nray,K,Nt,M,D,d_lamda=0.5,beta=1,std=10/180*np.pi):
    Ct = np.arange(Nt)
    H = np.zeros([M,K,Nt],dtype=complex)
    for k in range(K):
        Htemp = np.zeros([D,Nt],dtype=complex)
        for ii in range(Ncl):
            fhi_i = np.random.uniform(0,2*np.pi)
            for jj in range(Nray):
                a = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                fhi_ij = spreadAoD(fhi_i,std)
                ft = 1 / np.sqrt(Nt) * np.exp(Ct * 1j * 2 * np.pi * d_lamda * np.sin(fhi_ij))
                tao = np.random.uniform(0, D)
                for d in range(D):
                    Htemp[d,:] = Htemp[d,:] + a*ft*prc(d-tao,beta)
        Htemp = Htemp * np.sqrt(Nt/Ncl/Nray)
        for m in range(M):
            H[m,k,:] = np.matmul(np.exp(-1j*2*np.pi*m/M*np.arange(D)),Htemp)
    return H


def generate_H_dataset(number, K, N, M, D, Ncl, Nray):
    start = time.time()
    Hset = np.zeros([number, M, 2, K, N], dtype=np.float32)
    for i in range(number):
        if i % 2000 == 0:
            end = time.time()
            print(i, end - start)
            sys.stdout.flush()
            start = time.time()
        Htemp = MU_MISO_OFDM_MODEL(Ncl, Nray, K, Nt, M, D)
        Hset[i, :, 0, :, :] = np.real(Htemp)
        Hset[i, :, 1, :, :] = np.imag(Htemp)
    np.save("./data/setH_K" + str(K) + "_N" + str(N) + "_M" + str(M) + "_D" + str(D) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray) + "_number" + str(number) + ".npy", Hset)
    return Hset


if __name__ == '__main__':
    K = 3
    Nt = 16

    M = 8
    D = 3

    Ncl = 5
    Nray = 10

    # need a 'data' folder
    generate_H_dataset(500000, K, Nt, M, D, Ncl=Ncl, Nray=Nray)
    generate_H_dataset(10000, K, Nt, M, D, Ncl=Ncl, Nray=Nray)
