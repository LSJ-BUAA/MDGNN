# coding=UTF-8
import math
import torch
import torch.nn as nn


def complex_matmul(A,B):
    # the product of two complex matrices
    # A[b,m,2,h1,w1] B[b,m,2,h2,w2] output[b,2,h1,w2]
    b1, l1, c1, h1, w1 = A.size()
    return torch.cat([torch.matmul(A[:, :, 0, :, :], B[:, :, 0, :, :]) - torch.matmul(A[:, :, 1, :, :], B[:, :, 1, :, :]),
                      torch.matmul(A[:, :, 0, :, :], B[:, :, 1, :, :]) + torch.matmul(A[:, :, 1, :, :], B[:, :, 0, :, :])], dim=2).view(b1, l1, 2, h1, -1)


def ampli2(A):
    # the square amplitude of a complex matrix
    # A[b,m,2,h,w] output[b,m,h,w]
    return A[:,:,0,:,:]**2+A[:,:,1,:,:]**2


class Layer_4DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_4DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P5 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, permutation_size4, BATCH_SIZE, aggr_func=torch.mean):

        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3, permutation_size4])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim, permutation_size1, permutation_size2, permutation_size3, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim, permutation_size1, permutation_size2, 1, permutation_size4)
        A4 = torch.matmul(self.P4, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim, permutation_size1, 1, permutation_size3, permutation_size4)
        A5 = torch.matmul(self.P5, aggr_func(A, -4).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim, 1, permutation_size2, permutation_size3, permutation_size4)

        # A = A1 + A2 + A3 + A4 + A5

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1*A2 + 0.1*A3 + 0.1*A4 + 0.1*A5

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:  # the 4D-BN is realize by 1D-BN
            A = self.batch_norms(A.view([BATCH_SIZE, self.output_dim, -1])).view(
                    [BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3, permutation_size4])
        return A


class GNN4D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN4D, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_4DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_4DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False,is_transfer=False))

    def forward(self, A, M, K, Nt, Ns, equal_user, equal_subcarrier):
        #consider the permutations of subcarrier, users, and antennas
        BATCH_SIZE = self.BATCH_SIZE

        #input layer
        A = A.transpose(1, 2).contiguous()
        A = A.view(BATCH_SIZE,2,M,K,Nt,1)
        vfeature = torch.arange(-2, 2, 4 / Ns).view(1, 1, 1, 1, 1, Ns).to(self.device)
        A = A + vfeature

        #update layers
        for i in range(len(self.dim) - 1):
            A = self.layers[i](A, permutation_size1=M, permutation_size2=K,permutation_size3=Nt,permutation_size4=Ns, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0], dim=[1, 2]).transpose(1, 2)
        z2 = torch.mean(A[:, 1], dim=[1, 2]).transpose(1, 2)

        y1 = torch.mean(A[:, 2], dim=3)
        y2 = torch.mean(A[:, 3], dim=3)

        # to satisfy constraints
        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y = torch.stack([y1, y2], dim=2)
        z = torch.stack([z1, z2], dim=1).view([BATCH_SIZE, 1, 2, Ns,Nt])
        WT = complex_matmul(y, z)

        if equal_subcarrier:
            if equal_user:
                temp = math.sqrt(M*K)*torch.sqrt(torch.sum(ampli2(WT), dim=3)).view(
                    [BATCH_SIZE, M, 1, K, 1])
            else:
                temp = math.sqrt(M)*torch.sqrt(torch.sum(ampli2(WT), dim=[2, 3])).view(
                    [BATCH_SIZE, M, 1, 1, 1])
        else:
            if equal_user:
                temp = math.sqrt(K) * torch.sqrt(
                    torch.sum(ampli2(WT), dim=[1, 3])).view([BATCH_SIZE, 1, 1, K, 1])
            else:
                temp = torch.sqrt(torch.sum(ampli2(WT), dim=[1, 2, 3])).view([BATCH_SIZE, 1, 1, 1, 1])

        y = math.sqrt(M) * y / temp

        return y, z


class Layer_3DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_3DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm3d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0 / output_dim / input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, permutation_size3, BATCH_SIZE, aggr_func=torch.mean):


        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2, permutation_size3])
        A2 = torch.matmul(self.P2, aggr_func(A, -1).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim,permutation_size1,permutation_size2, 1)
        A3 = torch.matmul(self.P3, aggr_func(A, -2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim,permutation_size1,1, permutation_size3)
        A4 = torch.matmul(self.P4, aggr_func(A, -3).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE,self.output_dim,1,permutation_size2, permutation_size3)

        A = A1 + A2 + A3 + A4

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        # A = A1 + 0.1*A2 + 0.1*A3 + 0.1*A4

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN3D, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_3DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_3DPE(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False,is_transfer=False))

    def forward(self, A, M, K, Nt, Ns, equal_user, equal_subcarrier):
        #consider the permutations of subcarrier, users, and antennas
        BATCH_SIZE = self.BATCH_SIZE

        A = A.transpose(1, 2).contiguous()
        A = A.view(BATCH_SIZE,2,M,K,Nt)

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A, permutation_size1=M, permutation_size2=K,permutation_size3=Nt, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0:Ns], dim=[2, 3])
        z2 = torch.mean(A[:, Ns:2 * Ns], dim=[2, 3])
        y1 = torch.mean(A[:, 2 * Ns:3 * Ns], dim=4).transpose(1, 2).transpose(2, 3)
        y2 = torch.mean(A[:, 3 * Ns:], dim=4).transpose(1, 2).transpose(2, 3)

        # to satisfy constraints
        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y = torch.stack([y1, y2], dim=2)
        z = torch.stack([z1, z2], dim=1).view([BATCH_SIZE, 1, 2, Ns,Nt])
        WT = complex_matmul(y, z)

        if equal_subcarrier:
            if equal_user:
                temp = math.sqrt(M*K)*torch.sqrt(torch.sum(ampli2(WT), dim=3)).view(
                    [BATCH_SIZE, M, 1, K, 1])
            else:
                temp = math.sqrt(M)*torch.sqrt(torch.sum(ampli2(WT), dim=[2, 3])).view(
                    [BATCH_SIZE, M, 1, 1, 1])
        else:
            if equal_user:
                temp = math.sqrt(K) * torch.sqrt(
                    torch.sum(ampli2(WT), dim=[1, 3])).view([BATCH_SIZE, 1, 1, K, 1])
            else:
                temp = torch.sqrt(torch.sum(ampli2(WT), dim=[1, 2, 3])).view([BATCH_SIZE, 1, 1, 1, 1])

        y = math.sqrt(M) * y / temp

        return y, z


class Layer_2DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_2DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm2d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0/output_dim/input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2])
        A3 = torch.matmul(self.P3, aggr_func(A,-1).view(BATCH_SIZE,self.input_dim,-1)).view(BATCH_SIZE,self.output_dim,permutation_size1,1)
        A4 = torch.matmul(self.P4, aggr_func(A,-2).view(BATCH_SIZE,self.input_dim,-1)).view(BATCH_SIZE,self.output_dim,1,permutation_size2)

        # A = A1 + A3 + A4

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1*A3 + 0.1*A4

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN2D, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, A, M, K, Nt, Ns, equal_user, equal_subcarrier):
        # consider the permutations of subcarrier, antennas
        BATCH_SIZE = self.BATCH_SIZE

        A = A.transpose(1, 2).transpose(2, 3).contiguous()
        A = A.view(self.BATCH_SIZE, 2 * K, M, Nt)

        for i in range(len(self.dim)-1):
            A = self.layers[i](A, permutation_size1=M, permutation_size2=Nt, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0:Ns], dim=[2])
        z2 = torch.mean(A[:, Ns:2 * Ns], dim=[2])
        y1 = torch.mean(A[:, 2 * Ns:2 * Ns + K * Ns], dim=3).transpose(1, 2).view([BATCH_SIZE, M, K, Ns])
        y2 = torch.mean(A[:, 2 * Ns + K * Ns:], dim=3).transpose(1, 2).view([BATCH_SIZE, M, K, Ns])

        # to satisfy constraints
        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y = torch.stack([y1, y2], dim=2)
        z = torch.stack([z1, z2], dim=1).view([BATCH_SIZE, 1, 2, Ns, Nt])
        WT = complex_matmul(y, z)

        if equal_subcarrier:
            if equal_user:
                temp = math.sqrt(M * K) * torch.sqrt(torch.sum(ampli2(WT), dim=3)).view(
                    [BATCH_SIZE, M, 1, K, 1])
            else:
                temp = math.sqrt(M) * torch.sqrt(torch.sum(ampli2(WT), dim=[2, 3])).view(
                    [BATCH_SIZE, M, 1, 1, 1])
        else:
            if equal_user:
                temp = math.sqrt(K) * torch.sqrt(
                    torch.sum(ampli2(WT), dim=[1, 3])).view([BATCH_SIZE, 1, 1, K, 1])
            else:
                temp = torch.sqrt(torch.sum(ampli2(WT), dim=[1, 2, 3])).view([BATCH_SIZE, 1, 1, 1, 1])

        y = math.sqrt(M) * y / temp

        return y, z

