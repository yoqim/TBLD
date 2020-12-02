import os,sys
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.eval_metrics import ErrorRateAt95Recall
from src.uh_net import BrownFcModel
from src.uh_util import CalDistance,get_test_fea
from src.misc import *

import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.benchmark = True


def test(model,test_fea_a,test_fea_b,test_labels,test_batch_size):
    print("==> Testing ...")
    test_num = (test_fea_a.shape[0])
    batch_num = test_num // test_batch_size

    Fs_a = np.zeros((test_num,256))
    Fs_b = np.zeros((test_num,256))

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
                print(" mean : F_out_a {:.2f} & F_out_b {:.2f}".format(torch.mean(F_out_a),torch.mean(F_out_b)))

            B_out_a = torch.where(F_out_a<=0, -1*torch.ones_like(F_out_a), torch.ones_like(F_out_a))
            B_out_b = torch.where(F_out_b<=0, -1*torch.ones_like(F_out_b), torch.ones_like(F_out_b))

        Fs_a[start_idx:end_idx,:] = np.transpose(F_out_a.data.cpu().numpy())
        Fs_b[start_idx:end_idx,:] = np.transpose(F_out_b.data.cpu().numpy())

        dists_b = CalDistance(B_out_a, B_out_b, mode='h')
        distances_b[start_idx:end_idx] = dists_b.data.cpu().numpy()

    labels = np.asarray(test_labels)
    err95_b = ErrorRateAt95Recall(labels, distances_b)
    print(green('Test set: Accuracy Binary Code(ERR95): {:.2f}'.format(err95_b * 100)))

#### --- Draw F out distribution ---
    Fsa_mean = np.mean(Fs_a,axis=0)
    print("min Fsa_mean {:.3f}".format(np.min(Fsa_mean)))
    print("max Fsa_mean {:.3f}".format(np.max(Fsa_mean)))

    # Fsa_mean_abs = np.absolute(Fsa_mean)
    # c0 = 0
    # c1 = 0
    # c2 = 0
    # for fma in Fsa_mean_abs:
    #     if fma<0.51:
    #         c0 += 1
    #     elif fma>0.51 and fma<1.01:
    #         c1 += 1
    #     else:
    #         c2 += 1

    # c0 = c0/256
    # c1 = c1/256
    # c2 = c2/256
    # print("x < 0.5 : ",c0)
    # print("0.5 < x < 1 : ",c1)
    # print("x > 1 : ",c2)
    # import pdb;pdb.set_trace()

    plt.hist(Fsa_mean,bins=50,normed=1,facecolor='blue',histtype='stepfilled',alpha=0.4)
    plt.xlabel('Value')
    plt.ylabel('Percentage')
    # plt.xticks(np.arange(0, 3.5, step=0.5))
    # plt.yticks(np.arange(0, 1.2, step=0.2))
    # plt.title('Without ACM')
    # plt.savefig('./nogan.pdf', dpi=100)
    plt.title('With ACM')
    plt.savefig('./gan.pdf', dpi=100)
    import pdb;pdb.set_trace()


#### --- Save F out ---
    # save_path = '../../GraphBit/gencodes/nogan_{}2{}_bit.npy'.format(train_name[:3],test_name[:3])
    # np.save(save_path,Fs_a)
    # print("save to ",save_path)

    return err95_b


dataset_name = ['liberty', 'notredame', 'yosemite']
root_path = '/home/ym19/myq/TFeat/'
dataroot = root_path + 'data/BrownMat/'
train_name = dataset_name[2]
test_name = dataset_name[0]

G_bit = 1024  # real-value code length
F_bit = 256  # binary code length
test_batch_size = 500

model = BrownFcModel(G_bit=G_bit, F_bit=F_bit).cuda()
### --- WGAN code ---
model_dir = './model/'
test_model_name = 'yos2lib_wgp_aveneg_b21.95_epoch0.pth'
model.load_state_dict(torch.load(model_dir+test_model_name))

### --- For ContraCodes ---
# model_dir = '../ContraCodes/model/finalmodel/'
# test_model_name = 'yos2lib_G1024_F256_b22.07_epoch20.pth'
# tmp = torch.load(model_dir+test_model_name)
# tmp = tmp['model']
# model.load_state_dict(tmp)
# del tmp

model.eval()
print('=> Loading model {}'.format(test_model_name))

test_set = sio.loadmat(dataroot + '{}_multiview_test.mat'.format(test_name[0]))
test_set = test_set['view1_te']
match_file_path = root_path + 'data/Brown/{}/m50_10000_10000_0.txt'.format(test_name)
(test_fea_a,test_fea_b), test_labels = get_test_fea(test_set, match_file_path)

err95_b = test(model,test_fea_a,test_fea_b,test_labels,test_batch_size)
