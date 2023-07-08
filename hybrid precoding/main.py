# coding=UTF-8
import torch
import numpy as np
from random import shuffle
import time
import sys
from networks import complex_matmul, ampli2, GNN3D, GNN2D, GNN1D, AGNN2D


def cal_dataset_rate(H, BB, RF, K, BATCH_SIZE, SNR_dB, device):
    # compute the sum-rate of a batch
    # H(BATCH_SIZE,2,K,Nt), BB(BATCH_SIZE,2,K,Ns), RF(BATCH_SIZE,2,Ns,Nt)

    sigma2 = 10.0 ** (-SNR_dB / 10.0)
    W = complex_matmul(RF.transpose(2, 3), BB.transpose(2, 3))
    Q = complex_matmul(H, W)
    Q2 = ampli2(Q)
    D = torch.eye(K).to(device) * Q2
    sumD = torch.sum(D, dim=2, keepdim=True)
    sumQ = torch.sum(Q2, dim=2, keepdim=True)
    sinr = sumD/(sumQ-sumD+sigma2)
    rate = torch.sum(torch.log2(1.0 + sinr))
    return rate/BATCH_SIZE


if __name__ == '__main__':
    time_start = time.time()
    K = 3
    Ns = 6
    Nt = 16

    Ncl = 4
    Nray = 5

    SNR = 10

    set1_number = 500000
    set2_number = 10000

    train_number = 500000
    test_number = 10000

    H = np.load("./data/setH_K" + str(K) + "_N" + str(Nt) + "_Ncl" + str(Ncl) + "_Nray" + str(
        Nray) + "_number" + str(set1_number) + ".npy")

    Htest = np.load("./data/setH_K" + str(K) + "_N" + str(Nt) + "_Ncl" + str(Ncl) + "_Nray" + str(
        Nray) + "_number" + str(set2_number) + ".npy")

    # H = np.float32(np.random.randn(set1_number, 2, K, Nt) / np.sqrt(2))
    # Htest = np.float32(np.random.randn(set2_number, 2, K, Nt) / np.sqrt(2))

    BATCH_SIZE = 500

    # LEARNING_RATE = 1e-3
    LEARNING_RATE = 3e-4

    layer = [256]*5
    # layer = [128]*4
    # layer = [64]*4

    MAX_EPOCH = 1000

    device = torch.device("cuda:0")
    # device = torch.device("cpu")


    print('K',K,'Ns',Ns,'Nt',Nt,'Ncl',Ncl,'Nray',Nray,'SNR',SNR,'train_number',train_number,'testnumber',test_number)
    print('BatchSize',BATCH_SIZE,'LEARNING_RATE',LEARNING_RATE,'layer',layer)


    Xtrain = torch.from_numpy(H[0:train_number]).to(device)
    Xtest = torch.from_numpy(Htest[0:test_number]).to(device)


    # model = GNN3D(input_dim=2, hidden_dim=layer, output_dim=4, device=device,BATCH_SIZE=BATCH_SIZE)
    model = GNN2D(input_dim=2, hidden_dim=layer, output_dim=4*Ns, device=device, BATCH_SIZE=BATCH_SIZE)
    # model = GNN1D(input_dim=2*K, hidden_dim=layer, output_dim=2*K*Ns+2*Ns, device=device, BATCH_SIZE=BATCH_SIZE)
    # model = AGNN2D(input_dim=2, hidden_dim=layer, output_dim=4 * Ns, device=device, BATCH_SIZE=BATCH_SIZE)

    model.to(device)
    epoch_equal = 10 # in the first ten epoch, allocate same power to each user, in order to avoid a local optimum that only serve one/two users

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 170, 250, 350], gamma=0.3, last_epoch=-1)

    sys.stdout.flush()
    for epoch in range(MAX_EPOCH):
        index = [i for i in range(train_number)]
        shuffle(index)
        Xtrain = Xtrain[index]

        model.train()
        for b in range(int(train_number/BATCH_SIZE)):
            batch_x = Xtrain[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            if epoch < epoch_equal:
                y_pred, z_pred = model(batch_x, K, Nt, Ns, equal=True)
            else:
                y_pred, z_pred = model(batch_x, K, Nt, Ns, equal=False)
            loss = -cal_dataset_rate(H=batch_x, BB=y_pred, RF=z_pred, K=K, BATCH_SIZE=BATCH_SIZE, SNR_dB=SNR,device=device)

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
                    y_pred, z_pred = model(batch_x, K, Nt, Ns, equal=False)
                    loss = cal_dataset_rate(H=batch_x, BB=y_pred, RF=z_pred, K=K, BATCH_SIZE=BATCH_SIZE, SNR_dB=SNR,device=device)
                    sum_train_loss = sum_train_loss + loss.detach().cpu().numpy()

                    batch_x = Xtest[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                    y_pred, z_pred = model(batch_x, K, Nt, Ns, equal=False)
                    loss = cal_dataset_rate(H=batch_x, BB=y_pred, RF=z_pred, K=K, BATCH_SIZE=BATCH_SIZE, SNR_dB=SNR,device=device)
                    sum_test_loss = sum_test_loss + loss.detach().cpu().numpy()

                time_end = time.time()
                print(epoch,sum_train_loss/(test_number / BATCH_SIZE),sum_test_loss/(test_number / BATCH_SIZE),time_end-time_start)
                sys.stdout.flush()
                time_start = time.time()

        # if epoch % 100 == 0:
        #    torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, 'model.ckp')
