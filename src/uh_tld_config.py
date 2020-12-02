import warnings
import pickle
import scipy.io as sio
from src.uh_util import get_test_fea

dataset_name = ['liberty', 'notredame', 'yosemite']
best_fpr95_bs = [0.1722,0.37,0.2045,0.3574,0.22,0.21]

class DefaultConfig(object):
    root_path = '/home/ym19/myq/TFeat/'
    dataroot = root_path + 'data/BrownMat/'
    model_dir = './model/'
    
    train_name = dataset_name[0]
    test_name = dataset_name[1]

    resume = None
    # resume = model_dir+'lib2not_wgp_aveneg_b17.22_epoch30.pth'

    start_epoch = int(resume.split('_')[-1].strip('.pth').strip('epoch'))+1 if resume is not None else 0 
    total_epoch = 2000
    batch_size = 32
    test_batch_size = 1000
    gpu_id = '0'
    lr=1e-7

    G_bit = 1024  # real-value code length
    F_bit = 256  # binary code length

    scale = 0.1
    gamma = 2
    xwei = 1
    bwei = 1
    qwei = 1
    lamda = 1e-4

## gan loss paramters
    G_update_iter=10
    lambda_gp=10
    dlr=1e-5

    rotate_angles = [-10, -5, 5, 10]
    num_view = len(rotate_angles) + 1

    if train_name[:3] == 'lib':
        if test_name[:3] == 'not':
            best_fpr95_b = best_fpr95_bs[0]
        elif test_name[:3] == 'yos':
            best_fpr95_b = best_fpr95_bs[1]
    elif train_name[:3] == 'not':
        if test_name[:3] == 'lib':
            best_fpr95_b = best_fpr95_bs[2]
        elif test_name[:3] == 'yos':
            best_fpr95_b = best_fpr95_bs[3]
    elif train_name[:3] == 'yos':
        if test_name[:3] == 'lib':
            best_fpr95_b = best_fpr95_bs[4]
        elif test_name[:3] == 'not':
            best_fpr95_b = best_fpr95_bs[5]
    else:
        print("wrong train_name !!")

    tmp = sio.loadmat(dataroot + '{}_multiview_train.mat'.format(train_name[0]))
    train_set = {}
    train_set['0'] = tmp['view1_tr']
    for i, r in enumerate(rotate_angles):
        train_set[str(r)] = eval("tmp['view{}_tr']".format(i + 2))
    del tmp

    test_set = sio.loadmat(dataroot + '{}_multiview_test.mat'.format(test_name[0]))
    test_set = test_set['view1_te']
    match_file_path = root_path + 'data/Brown/{}/m50_10000_10000_0.txt'.format(test_name)
    test_set, test_labels = get_test_fea(test_set, match_file_path)

    n_clusters = 32
    with open('../ContraCodes/cluster/{}_{}_group.pickle'.format(train_name[:3],n_clusters), 'rb') as handle:
        clus_dict = pickle.load(handle)
    sel_neg_num = 1024

    # train2test
    log_interval = 10
    save_model_prefix = '{}2{}_tld_aveneg'.format(train_name[:3], test_name[:3])
    log_dir = './{}_lr{}_Dlr{}_lambda{}.txt'.format(save_model_prefix,lr,dlr,lamda)

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print((k, getattr(self, k)))


opt = DefaultConfig()
if __name__ == "__main__":
    import numpy as np

'''
##############
    Remove same feature vectors from the matrix
##############
    dataroot = opt.dataroot
    train_name = opt.train_name
    print("train name is {}".format(train_name))
    tmp = sio.loadmat(dataroot + '{}_multiview_train.mat'.format(train_name[0]))
    train_fea = tmp['view1_tr']
    train_num = train_fea.shape[0]
    indexs = np.arange(train_num)

    f= open('{}_pair_id.txt'.format(train_name[:3]),'w')
    a = set()
    for i in range(train_num-1):
        if i in a:
            continue
        cur_fea = train_fea[i,:]
        comp_fea = train_fea[i+1:,:]
        diff = list(np.sum(cur_fea-comp_fea,1))
        ind = [c+i+1 for c in range(len(diff)) if diff[c] == 0]
        if len(ind) > 0:
            print(ind)

            f.writelines("{} ".format(i))
            a.add(i)
            for j in ind:
                if j == ind[-1]:
                    f.writelines("{}\n".format(j))
                else:
                    f.writelines("{} ".format(j))
                a.add(j)

        if i%5000==0:
            print("to {}/{} line".format(i,train_num))
    print("the len of remove set {}".format(len(a)))

    f.close()

    with open('{}_remove_id.txt'.format(train_name[:3]),'w') as f:
        for i in a:
            f.writelines("{}.\n".format(i))

'''

'''
### check
    with open('{}_pair_id.txt'.format(train_name[:3]),'r') as f:
        aaa=f.readlines()


    for aa in aaa:
        ss = aa.strip('\n').split(' ')
        bbb = [np.sum(train_fea[int(i),:]) for i in ss]

        for i in range(1,len(bbb)):
            if bbb[0] != bbb[i]:
                print("wrong!")
'''
