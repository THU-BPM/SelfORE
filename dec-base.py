import os
import time
import pdb
from tqdm import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from metrics import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from parallel import DataParallelModel, DataParallelCriterion
import sys

CUDA = '1,2,3,4,5,6'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vec_len = 28 * 28
vec_len = 1536 # Bert input
cluster_num = None
# cluster_num = int(sys.argv[1])
print("=======DEC cluster num=====")
# print(cluster_num)
feature_num = 200
BATCH_SIZE = 16


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vec_len, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, feature_num))
            # nn.Linear(500, 500),
            # nn.ReLU(True),
            # nn.Linear(500, 2000),
            # nn.ReLU(True),
            # nn.Linear(2000, feature_num))
        self.decoder = nn.Sequential(
            # nn.Linear(feature_num, 2000),
            # nn.ReLU(True),
            # nn.Linear(2000, 500),
            # nn.ReLU(True),
            # nn.Linear(500, 500),
            nn.Linear(feature_num, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, vec_len))
        self.model = nn.Sequential(self.encoder, self.decoder)
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=cluster_num, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
            self.n_clusters,
            self.hidden,
            dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)
    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist

class DEC(nn.Module):
    def __init__(self, n_clusters=cluster_num, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x) 
        return self.clusteringlayer(x)

    def visualize(self, epoch,x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach() 
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:,0], x_embedded[:,1])
        fig.savefig('plots/mnist_{}.png'.format(epoch))
        plt.close(fig)

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        # print("=> Saving new checkpoint")
        torch.save(state, filename)
    # else:
    #     print("=> Validation Accuracy did not improve")

def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    parameters = list(autoencoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(dataset=data,
                    batch_size=BATCH_SIZE, 
                    shuffle=True)
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            img  = data.float()
            noisy_img = add_noise(img)
            noisy_img = img
            noisy_img = noisy_img.to(device)
            img = img.to(device)
            # ===================forward=====================
            output = model(noisy_img)
            output = output.squeeze(1)
            output = output.view(output.size(0), vec_len)
            loss = nn.MSELoss()(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], pretrain MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best': state,
                        'epoch':epoch
                        }, savepath,
                        is_best)


def train(**kwargs):
    data = kwargs['data']
    labels = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    features = []
    train_loader = DataLoader(dataset=data,
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    for i, batch in enumerate(train_loader):
        img = batch.float()
        # img.cuda()
        img = img.to(device)
        # features.append(model.autoencoder.encode(img).detach().cpu())
        features.append(model.module.autoencoder.encode(img).detach().cpu())
    features = torch.cat(features)
    print("Start KMeans")
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    # center_index = [7, 25, 1, 4, 0, 2, 24, 76, 6, 31]
    # cluster_centers = [features[idx] for idx in center_index]
    # cluster_centers = torch.stack(cluster_centers).cuda()

    model.module.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    print("End KMeans")
    # torch.cuda.empty_cache()
    # =========================================================
    # y_pred = kmeans.predict(features)
    # accuracy = acc(y.cpu().numpy(), y_pred)
    # print('Initial Accuracy: {}'.format(accuracy))

    loss_function = nn.KLDivLoss(size_average=False)
    # loss_function  = DataParallelCriterion(loss_function)
    optimizer = torch.optim.SGD(params=model.module.parameters(), lr=0.1, momentum=0.9)
    # optimizer = torch.optim.Adam(params=model.module.parameters(), lr=1e-2, weight_decay=1e-5)
    print('Training')
    row = []
    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float()
        # img = img.to(device)
        output = model(img)
        output=output.cpu()
        target = model.module.target_distribution(output).detach()
        out = output.argmax(1)
        
        #out = output.argmax(1)
        # if epoch % 20 == 0:
        #     print('plotting')
        #     dec.visualize(epoch, img)
        loss = loss_function(output.log(), target) / output.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
        # print("OUT", out[:10], len(out), type(out))
        # row.append([epoch, accuracy])
        # print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch, num_epochs, accuracy, loss))
        state = loss.item()
        is_best = False
        if epoch % 20 == 0:
            print("Epoch {} Loss: {}".format(epoch, state))
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True
            print("Epoch {} now is best DEC, saved".format(epoch))
            import pickle
            
            # import ipdb; ipdb.set_trace() 

            # kmeans_output = KMeans(n_clusters=100, random_state=0).fit(output.detach().numpy())
            # with open('DEC_out_100', 'wb') as fp:
            # with open('DEC_out_'+str(cluster_num), 'wb') as fp:
            #     out_save = kmeans_output.labels_.tolist()
            #     pickle.dump(out_save, fp)
    
  
            with open('DEC_out_'+str(cluster_num), 'wb') as fp:
               pickle.dump(out.tolist(), fp)

        save_checkpoint({
                        'state_dict': model.module.state_dict(),
                        'best': state,
                        'epoch':epoch
                        }, savepath,
                        is_best)

    # import ipdb; ipdb.set_trace()
    print("Train End")
    # df = pd.DataFrame(row, columns=['epochs', 'accuracy'])
    # df.to_csv('log.csv')


def load_mnist():
    # the data, shuffled and split between train and test sets
    train = MNIST(root='./data/',
                train=True, 
                transform=transforms.ToTensor(),
                download=True)

    test = MNIST(root='./data/',
                train=False, 
                transform=transforms.ToTensor())
    x_train, y_train = train.train_data, train.train_labels
    x_test, y_test = test.test_data, test.test_labels
    x = torch.cat((x_train, x_test), 0)
    y = torch.cat((y_train, y_test), 0)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y

def load_bert():        
    print("load bert")
    import pickle
    with open ('BERT_out_'+str(cluster_num), 'rb') as fp:
        lst = pickle.load(fp)

    from numpy import array
    lst_np = array(lst)
    x = torch.from_numpy(lst_np)
    y = torch.zeros(x.shape[0], dtype=torch.int32)
    print('Bert Input', x.shape)
    return x, y
    

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--pretrain_epochs', default=20, type=int) # set to 10, faster
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='saves')
    parser.add_argument('--cluster_num', default=800, type=int)
    args = parser.parse_args()
    print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size
    cluster_num = args.cluster_num
    print("===Cluster Num===")
    print(cluster_num)

    # x, y = load_mnist()
    x, y = load_bert()
    autoencoder = AutoEncoder().to(device)
    ae_save_path = 'saves/sim_autoencoder.pth'

    import subprocess
    cmd = "rm saves/sim_autoencoder.pth saves/dec.pth"
    subprocess.getstatusoutput(cmd)


    # if os.path.isfile(ae_save_path):
    #     print('Loading {}'.format(ae_save_path))
    #     checkpoint = torch.load(ae_save_path)
    #     autoencoder.load_state_dict(checkpoint['state_dict'])
    # else:
    print("=> no checkpoint found at '{}'".format(ae_save_path))
    checkpoint = {
        "epoch": 0,
        "best": float("inf")
    }
    pretrain(data=x, model=autoencoder, num_epochs=epochs_pre, savepath=ae_save_path, checkpoint=checkpoint)


    dec_save_path='saves/dec.pth'
    dec = DEC(n_clusters=cluster_num, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0)
    #torch.distributed.init_process_group(backend="nccl")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #dec = torch.nn.parallel.DistributedDataParallel(dec)
        # dec = DataParallelModel(dec)
        dec = nn.DataParallel(dec)
    dec.to(device)


    # if os.path.isfile(dec_save_path):
    #     print('Loading {}'.format(dec_save_path))
    #     checkpoint = torch.load(dec_save_path)
    #     dec.load_state_dict(checkpoint['state_dict'])
    # else:
    print("=> no checkpoint found at '{}'".format(dec_save_path))
    checkpoint = {
        "epoch": 0,
        "best": float("inf")
    }
    train(data=x, labels=y, model=dec, num_epochs=args.train_epochs, savepath=dec_save_path, checkpoint=checkpoint)
