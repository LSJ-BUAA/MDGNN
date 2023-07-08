# coding=UTF-8
import torch
import torch.nn as nn
import numpy as np
from random import shuffle
import time
import sys


def cal_dataset_rate(H, p, K, BATCH_SIZE, SNR_dB, device):
    # compute the sum-rate of a batch
    # H(BATCH_SIZE,K,K), p(BATCH_SIZE,K)
    number = BATCH_SIZE
    s2 = 10.0 ** (-SNR_dB / 10.0)
    H = p.reshape([number, 1, K]) * H
    D = torch.eye(K).to(device) * H
    sumD = torch.sum(D, 2)
    sumQ = torch.sum(H, 2)
    sinr = sumD / (sumQ - sumD + s2)
    rate = torch.log2(1.0 + sinr)
    return torch.sum(rate) / number


class GNN2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(GNN2D, self).__init__()

        self.batch_norms = torch.nn.ModuleList()
        self.P1 = nn.ParameterList()
        self.P2 = nn.ParameterList()
        self.P3 = nn.ParameterList()
        self.P4 = nn.ParameterList()
        self.P5 = nn.ParameterList()
        self.P6 = nn.ParameterList()
        self.P7 = nn.ParameterList()
        self.P8 = nn.ParameterList()

        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 2):
            self.batch_norms.append(nn.BatchNorm2d(self.dim[i + 1]))
        for i in range(len(self.dim) - 1):
            ini = 3.0 / torch.sqrt(torch.FloatTensor([self.dim[i+1] * self.dim[i]]))
            self.P1.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P2.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P3.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P4.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P5.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P6.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P7.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))
            self.P8.append(nn.Parameter(torch.rand([self.dim[i + 1], self.dim[i]], requires_grad=True) * 2 * ini - ini))

        self.activation = nn.ReLU()
        self.device = device
        self.out = nn.Sigmoid()

        self.E = torch.eye(K).to(device)
        self.F = torch.ones([K,K]).to(device)
        self.G = self.F-self.E

        self.Q1 = nn.Parameter(torch.rand([], requires_grad=True) * 2 * ini - ini)
        self.Q2 = nn.Parameter(torch.rand([], requires_grad=True) * 2 * ini - ini)
        self.Q3 = nn.Parameter(torch.rand([], requires_grad=True) * 2 * ini - ini)
        self.Q4 = nn.Parameter(torch.rand([], requires_grad=True) * 2 * ini - ini)
        self.Q5 = nn.Parameter(torch.rand([], requires_grad=True) * 2 * ini - ini)

    def forward(self, A):

        BATCH_SIZE,K,_ = A.shape
        A = A.unsqueeze(1)

        for i in range(len(self.P1)):
            A1 = torch.matmul(self.P1[i],(self.E*A).view([BATCH_SIZE,self.dim[i],-1])).view([BATCH_SIZE,self.dim[i+1],K,K])
            A2 = self.E*torch.matmul(self.P2[i],torch.sum(A,-1).view([BATCH_SIZE,self.dim[i],-1])).view([BATCH_SIZE,self.dim[i+1],K,1])
            A3 = self.E * torch.matmul(self.P3[i], torch.sum(A, -2).view([BATCH_SIZE, self.dim[i], -1])).view([BATCH_SIZE, self.dim[i + 1], 1, K])

            A4 = torch.matmul(self.P4[i],(self.G*A).view([BATCH_SIZE,self.dim[i],-1])).view([BATCH_SIZE,self.dim[i+1],K,K])
            A5 = self.G*torch.matmul(self.F,torch.matmul(self.P5[i],(self.E*A).view([BATCH_SIZE,self.dim[i],-1])).view([BATCH_SIZE,self.dim[i+1],K,K]))
            A6 = self.G*torch.matmul(torch.matmul(self.P6[i],(self.E*A).view([BATCH_SIZE,self.dim[i],-1])).view([BATCH_SIZE,self.dim[i+1],K,K]),self.F)

            A7 = self.G * torch.matmul(self.P7[i],torch.sum(A,-1).view([BATCH_SIZE,self.dim[i],-1])).view([BATCH_SIZE,self.dim[i+1],K,1])
            A8 = self.G * torch.matmul(self.P8[i], torch.sum(A, -2).view([BATCH_SIZE, self.dim[i], -1])).view([BATCH_SIZE, self.dim[i + 1], 1, K])

            A = A1 + 0.1*A2 + 0.1*A3 + A4 + 0.1*A5 + 0.1*A6 + 0.01*A7 + 0.01*A8

            if i != len(self.P1) - 1:
                A = self.activation(A)
                A = self.batch_norms[i](A)
                # A = self.activation(A)

        #output layer
        y = self.Q1*torch.diagonal(A,dim1=-1,dim2=-2).view([BATCH_SIZE, K])\
            +0.1*self.Q2*torch.sum(A,-1).view([BATCH_SIZE, K])\
            +0.1*self.Q3*torch.sum(A,-2).view([BATCH_SIZE, K])\
            +0.05*self.Q4* torch.sum(torch.diagonal(A,dim1=-1,dim2=-2),-1)\
            +0.01*self.Q5 * torch.sum(A,[-1,-2])
        y = self.out(y)

        return y


if __name__ == '__main__':
    time_start = time.time()
    K = 20
    SNR = 10

    train_number = 10000
    test_number = 2000

    H = 1 / np.sqrt(2) * (np.random.randn(train_number,K, K) + 1j * np.random.randn(train_number,K, K))
    H = np.float32(np.abs(H)**2)

    Htest = 1 / np.sqrt(2) * (np.random.randn(test_number,K, K) + 1j * np.random.randn(test_number,K, K))
    Htest = np.float32(np.abs(Htest)**2)

    MAX_EPOCH = 1000
    device = torch.device("cuda:0")

    Xtrain = torch.from_numpy(H[0:train_number]).to(device)
    Xtest = torch.from_numpy(Htest[0:test_number]).to(device)

    BATCH_SIZE = 50
    LEARNING_RATE = 3e-4

    # layer = [512] * 6
    layer = [256] * 6
    # layer = [128] * 6
    # layer = [64] * 4

    model = GNN2D(input_dim=1, hidden_dim=layer, output_dim=1, device=device)
    model.to(device)

    print('K',K,'SNR',SNR,'train_number',train_number,'test_number',test_number,'BS',BATCH_SIZE,'LR',LEARNING_RATE,'layer',layer)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 170, 250, 300], gamma=0.3, last_epoch=-1)

    sys.stdout.flush()
    for epoch in range(MAX_EPOCH):
        index = [i for i in range(train_number)]
        shuffle(index)
        Xtrain = Xtrain[index]

        model.train()
        for b in range(int(train_number / BATCH_SIZE)):
            batch_x = Xtrain[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            y_pred = model(batch_x)
            loss = -cal_dataset_rate(H=batch_x, p=y_pred, K=K, BATCH_SIZE=BATCH_SIZE, SNR_dB=SNR, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                sum_train_loss = 0
                sum_test_loss = 0
                for b in range(int(test_number / BATCH_SIZE)):
                    batch_x = Xtrain[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                    y_pred = model(batch_x)
                    loss = cal_dataset_rate(H=batch_x, p=y_pred, K=K, BATCH_SIZE=BATCH_SIZE, SNR_dB=SNR,device=device)
                    sum_train_loss = sum_train_loss + loss.detach().cpu().numpy()

                    batch_x = Xtest[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                    y_pred = model(batch_x)
                    loss = cal_dataset_rate(H=batch_x, p=y_pred, K=K, BATCH_SIZE=BATCH_SIZE, SNR_dB=SNR,device=device)
                    sum_test_loss = sum_test_loss + loss.detach().cpu().numpy()

                time_end = time.time()
                print(epoch, sum_train_loss / (test_number / BATCH_SIZE), sum_test_loss / (test_number / BATCH_SIZE),time_end - time_start)
                sys.stdout.flush()
                time_start = time.time()