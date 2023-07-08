# coding=UTF-8
import math
import torch
import torch.nn as nn


def complex_matmul(A,B):
    # the product of two complex matrices
    # A[b,2,h1,w1] B[b,2,h2,w2] output[b,2,h1,w2]
    b1, _, h1, _ = A.size()
    return torch.cat([torch.matmul(A[:, 0, :, :],B[:, 0, :, :])-torch.matmul(A[:, 1, :, :],B[:, 1, :, :]),
                      torch.matmul(A[:, 0, :, :],B[:, 1, :, :])+torch.matmul(A[:, 1, :, :],B[:, 0, :, :])],dim=1).view(b1,2,h1,-1)


def ampli2(A):
    # the square amplitude of a complex matrix
    # A[b,2,h,w] output[b,h,w]
    return A[:,0,:,:]**2+A[:,1,:,:]**2


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

    def forward(self, A, K, Nt, Ns, equal):
        #consider the permutations of users, antennas, and RFchains
        BATCH_SIZE = self.BATCH_SIZE
        # the K,Nt,BATCH_SIZE can be different in training and testing, input a new K,Nt,BATCH_SIZE or use BATCH_SIZE,_,K,Nt=A.shape

        vfeature = torch.arange(-2, 2, 4 / Ns).view(1, 1, 1, 1, Ns).to(self.device) # input layer: the Virtual Feature
        A = A.view(BATCH_SIZE,2,K,Nt,1)+vfeature # input layer

        for i in range(len(self.dim) - 1):
            A = self.layers[i](A, permutation_size1=K, permutation_size2=Nt,permutation_size3=Ns, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0], dim=1).transpose(1, 2)
        z2 = torch.mean(A[:, 1], dim=1).transpose(1, 2)
        y1 = torch.mean(A[:, 2], dim=2)
        y2 = torch.mean(A[:, 3], dim=2)

        # to satisfy constraints
        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y1 = y1.view([BATCH_SIZE, 1, K, Ns])
        y2 = y2.view([BATCH_SIZE, 1, K, Ns])
        z1 = z1.view([BATCH_SIZE, 1, Ns, Nt])
        z2 = z2.view([BATCH_SIZE, 1, Ns, Nt])
        y = torch.cat([y1, y2], dim=1)
        z = torch.cat([z1, z2], dim=1)
        WT = complex_matmul(y, z)

        if equal:  # equal power for each user
            temp = math.sqrt(K) * torch.sqrt(torch.sum(ampli2(WT), dim=2)).view([BATCH_SIZE, 1, K, 1])
        else:
            temp = torch.sqrt(torch.sum(ampli2(WT), dim=[1, 2])).view([BATCH_SIZE, 1, 1, 1])
        y = y / temp
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

    def forward(self, A, K, Nt, Ns, equal):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        # the K,Nt,BATCH_SIZE can be different in training and testing, input a new K,Nt,BATCH_SIZE or use BATCH_SIZE,_,K,Nt=A.shape

        for i in range(len(self.dim)-1):
            A = self.layers[i](A, permutation_size1=K, permutation_size2=Nt, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0:Ns], dim=2)
        z2 = torch.mean(A[:, Ns:2*Ns], dim=2)
        y1 = torch.mean(A[:, 2*Ns:3*Ns], dim=3).transpose(1,2)
        y2 = torch.mean(A[:, 3*Ns:], dim=3).transpose(1,2)

        # to satisfy constraints
        mo = torch.sqrt(z1**2 + z2**2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y1 = y1.view([BATCH_SIZE, 1, K, Ns])
        y2 = y2.view([BATCH_SIZE, 1, K, Ns])
        z1 = z1.view([BATCH_SIZE, 1, Ns, Nt])
        z2 = z2.view([BATCH_SIZE, 1, Ns, Nt])
        y = torch.cat([y1, y2], dim=1)
        z = torch.cat([z1, z2], dim=1)
        WT = complex_matmul(y,z)

        if equal: # equal power for each user
            temp = math.sqrt(K)*torch.sqrt(torch.sum(ampli2(WT),dim=2)).view([BATCH_SIZE,1,K,1])
        else:
            temp = torch.sqrt(torch.sum(ampli2(WT),dim=[1,2])).view([BATCH_SIZE,1,1,1])
        y = y/temp
        return y, z


class Layer_1DPE(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_1DPE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0/output_dim/input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)


    def forward(self, A, permutation_size1, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A)
        A2 = torch.matmul(self.P2, aggr_func(A,-1,keepdim=True))

        # A = A1 + A2
        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        A = A1 + 0.1*A2

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class GNN1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(GNN1D, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_1DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_1DPE(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, A, K, Nt, Ns, equal):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        # the K,Nt,BATCH_SIZE can be different in training and testing, input a new K,Nt,BATCH_SIZE or use BATCH_SIZE,_,K,Nt=A.shape

        A = A.view(BATCH_SIZE, 2 * K, Nt) # input layer

        for i in range(len(self.dim)-1):
            A = self.layers[i](A, permutation_size1=Nt, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = A[:, 0:Ns]
        z2 = A[:, Ns:2 * Ns]
        y1 = torch.mean(A[:, 2 * Ns:2 * Ns + K * Ns], dim=2)
        y2 = torch.mean(A[:, 2 * Ns + K * Ns:2 * Ns + 2 * K * Ns], dim=2)

        # to satisfy constraints
        mo = torch.sqrt(z1**2 + z2**2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y1 = y1.view([BATCH_SIZE, 1, K, Ns])
        y2 = y2.view([BATCH_SIZE, 1, K, Ns])
        z1 = z1.view([BATCH_SIZE, 1, Ns, Nt])
        z2 = z2.view([BATCH_SIZE, 1, Ns, Nt])
        y = torch.cat([y1, y2], dim=1)
        z = torch.cat([z1, z2], dim=1)
        WT = complex_matmul(y,z)

        if equal: # equal power for each user
            temp = math.sqrt(K)*torch.sqrt(torch.sum(ampli2(WT),dim=2)).view([BATCH_SIZE,1,K,1])
        else:
            temp = torch.sqrt(torch.sum(ampli2(WT),dim=[1,2])).view([BATCH_SIZE,1,1,1])
        y = y/temp
        return y, z


class Layer_2DPE_attention(nn.Module):
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True):
        super(Layer_2DPE_attention, self).__init__()

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

        self.q = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.k = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.v = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)


    def forward(self, A, permutation_size1, permutation_size2, BATCH_SIZE, aggr_func=torch.mean):
        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view([BATCH_SIZE, self.output_dim, permutation_size1, permutation_size2])
        A3 = torch.matmul(self.P3, aggr_func(A,-1).view(BATCH_SIZE,self.input_dim,-1)).view(BATCH_SIZE,self.output_dim,permutation_size1,1)

        AT = A.transpose(1,2)

        Q = torch.matmul(self.q,AT).transpose(1,2)

        K = torch.matmul(self.k,AT)
        KT = K.permute(0,2,3,1)

        V = torch.matmul(self.v,AT).transpose(1,2)

        Alpha = torch.matmul(Q,KT)/permutation_size2
        Y = torch.matmul(Alpha,V)

        A = A1 + A3 + Y

        # if aggr_func=torch.sum or difficult to converge, because aggregated information is too large compared to itself
        # A =

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A)
        return A


class AGNN2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, BATCH_SIZE):
        super(AGNN2D, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.BATCH_SIZE = BATCH_SIZE
        self.device = device
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(Layer_2DPE_attention(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True))
            else:
                self.layers.append(Layer_2DPE_attention(self.dim[i], self.dim[i+1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False))

    def forward(self, A, K, Nt, Ns, equal):
        # consider the permutations of users, antennas
        BATCH_SIZE = self.BATCH_SIZE
        # the K,Nt,BATCH_SIZE can be different in training and testing, input a new K,Nt,BATCH_SIZE or use BATCH_SIZE,_,K,Nt=A.shape

        for i in range(len(self.dim)-1):
            A = self.layers[i](A, permutation_size1=K, permutation_size2=Nt, BATCH_SIZE=BATCH_SIZE)

        # output layer: y for W_BB, z for W_RF
        z1 = torch.mean(A[:, 0:Ns], dim=2)
        z2 = torch.mean(A[:, Ns:2*Ns], dim=2)
        y1 = torch.mean(A[:, 2*Ns:3*Ns], dim=3).transpose(1,2)
        y2 = torch.mean(A[:, 3*Ns:], dim=3).transpose(1,2)

        # to satisfy constraints
        mo = torch.sqrt(z1**2 + z2**2)
        z1 = z1 / mo
        z2 = z2 / mo

        # to satisfy constraints
        y1 = y1.view([BATCH_SIZE, 1, K, Ns])
        y2 = y2.view([BATCH_SIZE, 1, K, Ns])
        z1 = z1.view([BATCH_SIZE, 1, Ns, Nt])
        z2 = z2.view([BATCH_SIZE, 1, Ns, Nt])
        y = torch.cat([y1, y2], dim=1)
        z = torch.cat([z1, z2], dim=1)
        WT = complex_matmul(y,z)

        if equal: # equal power for each user
            temp = math.sqrt(K)*torch.sqrt(torch.sum(ampli2(WT),dim=2)).view([BATCH_SIZE,1,K,1])
        else:
            temp = torch.sqrt(torch.sum(ampli2(WT),dim=[1,2])).view([BATCH_SIZE,1,1,1])
        y = y/temp
        return y, z