import os

import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init
from torch.optim import SGD
import random
from torch.distributions.bernoulli import Bernoulli

sys.path.append('./src/')
import numpy as np

from src.eval_metrics import ErrorRateAt95Recall
from src.uh_tld_config import opt
from src.uh_net import BrownFcModel,Discriminator
from src.uh_util import CalDistance, get_batch_data, CompRealCode,SelectNegByCluster
from src.uh_tld_loss import CountLoss, calc_gradient_penalty,UpdateAlpha
from src.misc import *

dataroot = opt.dataroot
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
opt.cuda = torch.cuda.is_available()
total_epoch = opt.total_epoch
start_epoch = opt.start_epoch
gamma = opt.gamma
scale = opt.scale
num_view = opt.num_view
train_fea = opt.train_set
batch_size = opt.batch_size
test_batch_size = opt.test_batch_size
G_bit = opt.G_bit
F_bit = opt.F_bit
cudnn.benchmark = True if opt.cuda else False
best_model_name_prefix ="None"
LOG_DIR = opt.log_dir
if LOG_DIR is not None:
    print("log writing to " + LOG_DIR)
    f = open(LOG_DIR, 'w+')
loss_weights = [opt.xwei, opt.bwei, opt.qwei, opt.lamda]

## wgp loss paramters
G_update_iter=opt.G_update_iter
lambda_gp=opt.lambda_gp
dlr=opt.dlr

## cluster params
clus_dict = opt.clus_dict
sel_neg_num = opt.sel_neg_num # if<0: all neg

def get_gan_Fbit(train_fea,model,tr_index,batch_num,batch_size,pow_alpha):
    gan_start_idx = random.randint(0,(batch_num-1)*batch_size)
    gan_indecies = tr_index[gan_start_idx:gan_start_idx+batch_size]
    cur_gan_fea = get_batch_data(train_fea, gan_indecies)
    Xi_, Bi_ = NetForward(cur_gan_fea, model)

    feat = CompRealCode(Bi_,pow_alpha)
    feat = feat.t()

    norm = feat.norm(p=2, dim=1, keepdim=True)
    fea = feat/norm

    return fea


def NetForward(data, model):
    Xi_ = []
    Fi_ = []

    for _, (_, fea) in enumerate(data.items()):
        fea = torch.from_numpy(fea).cuda()
        X_out, F_out = model(fea)

        Xi_.append(X_out)
        Fi_.append(F_out)

    return Xi_, Fi_


def train(train_fea, model, X, B, optimizer, Discriminator, D_optim, epoch, pow_alpha):
    print("==> Training ...")

    permutation = [i for i in range(train_num)]
    np.random.shuffle(permutation)

    batch_num = train_num // batch_size
    print("train batch num {}".format(batch_num))

    for batch_idx in range(batch_num):
        ### ------ train discriminator ------
        F_gan_D = get_gan_Fbit(train_fea,model,permutation,batch_num,batch_size,pow_alpha)

        binary_code = torch.bernoulli(torch.rand(batch_size, F_bit)).cuda()
        binary_code = torch.where(binary_code==0, -1*torch.ones_like(binary_code), torch.ones_like(binary_code))
        gradient_penalty = calc_gradient_penalty(D, binary_code.data, F_gan_D.data)

        errD_fake = D(F_gan_D)
        errD_fake = errD_fake.mean()
        errD_real = D(binary_code)
        errD_real = errD_real.mean()

        D_optim.zero_grad()
        lossD=-errD_real + errD_fake + gradient_penalty * lambda_gp
        lossD.backward()
        D_optim.step()

        del F_gan_D
        del errD_real, errD_fake
        # print("Batch idx {}: lossD {:.3f}".format(batch_idx, lossD.data))

        ### ------ train generator ------
        if batch_idx % G_update_iter == 0:
            optimizer.zero_grad()

            start_idx = batch_size * batch_idx
            end_idx = (start_idx + batch_size) if batch_idx < batch_num - 1 else train_num
            indecies = permutation[start_idx:end_idx]
            if sel_neg_num>0:
                sel_neg_indecies = SelectNegByCluster(sel_neg_num,clus_dict,indecies)
            else:
                sel_neg_indecies = None

            cur_tr_fea = get_batch_data(train_fea, indecies)
            Xi_, Bi_ = NetForward(cur_tr_fea, model)
            del cur_tr_fea
            loss,loss_parts = CountLoss(Xi_, Bi_, X, B, loss_weights, scale, indecies, batch_idx, sel_neg_indecies)
            
            feat = CompRealCode(Bi_,pow_alpha)
            feat = feat.t()
            norm = feat.norm(p=2, dim=1, keepdim=True)
            norm_fea = feat/norm
            
            errG = D(norm_fea)
            lossG = loss-errG.mean()
            lossG.backward()
            optimizer.step()

            alpha = UpdateAlpha(loss_parts,loss_weights,gamma)
            pow_alpha = torch.pow(alpha, gamma)
            del loss_parts

        if batch_idx % (G_update_iter*60) == 0:
            print("Batch idx {}: lossG: {:.2f}, lossD: {:.2f}".format(batch_idx, lossG.data, lossD.data))
            print("new alpha {}".format(alpha.data))
    return pow_alpha


def test(model, test_batch_size):
    print("==> Testing ...")
    model.eval()

    (test_fea_a, test_fea_b), test_labels = opt.test_set, opt.test_labels

    test_num = (test_fea_a.shape[0])
    batch_num = test_num // test_batch_size

    distances_b = np.zeros(test_num)
    for batch_idx in range(batch_num):
        start_idx = batch_idx * test_batch_size
        end_idx = (start_idx + test_batch_size) if batch_idx < batch_num - 1 else test_num
        cur_fea_a = torch.from_numpy(test_fea_a[start_idx:end_idx, :]).cuda()
        cur_fea_b = torch.from_numpy(test_fea_b[start_idx:end_idx, :]).cuda()

        with torch.no_grad():
            X_out_a, F_out_a = model(cur_fea_a)
            X_out_b, F_out_b = model(cur_fea_b)
            if batch_idx % 100 ==0:
                print("mean: F_out_a {:.2f} & F_out_b {:.2f}".format(torch.mean(F_out_a),torch.mean(F_out_b)))

            B_out_a = torch.where(F_out_a<=0, -1*torch.ones_like(F_out_a), torch.ones_like(F_out_a))
            B_out_b = torch.where(F_out_b<=0, -1*torch.ones_like(F_out_b), torch.ones_like(F_out_b))

        dists_b = CalDistance(B_out_a, B_out_b, mode='h')
        distances_b[start_idx:end_idx] = dists_b.data.cpu().numpy()

    labels = np.asarray(test_labels)
    err95_b = ErrorRateAt95Recall(labels, distances_b)

    print(green('Test set: Accuracy Binary Code(ERR95): {:.2f}'.format(err95_b * 100)))
    return err95_b

## Hash Model
model = BrownFcModel(G_bit=G_bit, F_bit=F_bit).cuda()
optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

## Discriminator
D = Discriminator(F_bit).cuda()
D_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, D.parameters()), lr=1e-8,
                            momentum=0.9,
                            weight_decay=5e-4,
                            nesterov=True)


if opt.resume:
    if os.path.isfile(opt.resume):
        print('=> loading checkpoint {}'.format(opt.resume))
        tmp = torch.load(opt.resume)
        model.load_state_dict(tmp['model'])
        alpha = tmp['alpha']
    else:
        print('=> no checkpoint found at {}'.format(opt.resume))
else:
    print('=> train from scratch...')
    alpha = torch.ones(num_view) / num_view

pow_alpha = torch.pow(alpha, gamma)

## remove same samples
remove_set = []
with open('../{}_reid.txt'.format(opt.save_model_prefix.split('_')[0].split('2')[0]), 'r') as fin:
    for line in fin.readlines():
        line = line.strip('\n')
        remove_set.append(int(line))
remain_set = [i for i in range(train_fea['0'].shape[0]) if i not in remove_set]
if (len(remain_set)%2)!=0:
    remain_set = remain_set[:-1]
train_num = len(remain_set)

for key in train_fea.keys():
    train_fea[key] = train_fea[key][remain_set, :]
#######################

for epoch in range(start_epoch,total_epoch):
    if epoch == start_epoch:
        best_fpr95_b = opt.best_fpr95_b

    model.eval()
    with torch.no_grad():
        Xi_, Bi_ = NetForward(train_fea, model)
        F = CompRealCode(Bi_,pow_alpha)
        B = torch.where(F<=0, -1*torch.ones_like(F), torch.ones_like(F))
        # tmp_binary_norm = 1 / np.sqrt(F_bit * (0.5**2)) * torch.ones_like(F).cuda()
        # B = torch.where(F>=tmp_binary_norm/2, -1*torch.ones_like(F), torch.ones_like(F))

        # nums = [''.join(['{}'.format(j) for j in B[:, i].int().cpu().numpy()]) for i in range(B.shape[1])]
        # numset = set(nums)
        # if B.shape[1]-len(numset) > 0:
        #     print(red('same hash code {}'.format(B.shape[1]-len(numset))))

        X = CompRealCode(Xi_, pow_alpha)

    model.train()
    pow_alpha=train(train_fea, model, X, B, optimizer, D, D_optim, epoch, pow_alpha)
    print("epoch {}, pow_alpha {}".format(epoch,pow_alpha))

    err95_b = test(model,test_batch_size)
    if err95_b < best_fpr95_b:
        best_model_name_prefix = opt.save_model_prefix + '_b{:.2f}_epoch{}'.format(err95_b * 100, epoch)
        # torch.save(model.state_dict(), opt.model_dir + best_model_name_prefix+'.pth')
        torch.save({'model':model.state_dict(),'alpha':alpha.data}, opt.model_dir + best_model_name_prefix + '.pth')
    best_fpr95_b = min(err95_b, best_fpr95_b)
    print(green("---- Epoch {} ----".format(epoch)))
    print(green("=> best model name : "+ best_model_name_prefix))
    print(green("=> best_fpr95_b : {:.2f}%".format(best_fpr95_b*100)))

    if LOG_DIR:
        f.writelines("==> Epoch {}\n".format(epoch))
        f.writelines("Binary fpr95: {:.3f}\n".format(err95_b * 100))


if LOG_DIR:
    f.close()
