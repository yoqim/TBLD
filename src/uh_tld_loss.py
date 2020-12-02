import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss
from .misc import *
import torch.autograd as autograd
#import pdb; pdb.set_trace()

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates.requires_grad = True
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
    return gradient_penalty

def PartXmulloss(Xi,X_neg):
    batch_size = Xi.size(1)
    sam_size = X_neg.size(1)
    dist_map_0 = Xi.pow(2).sum(dim=0, keepdim=True).expand(sam_size, batch_size) + \
                X_neg.pow(2).sum(dim=0, keepdim=True).expand(batch_size, sam_size).t()
    dist_map_0 = dist_map_0.t()
    dist_map_0.addmm_(1, -2, Xi.t(), X_neg)
    dist_map_0 = torch.sqrt(dist_map_0.mean(dim=1))
    return dist_map_0

def CalXmulloss(Xi,X,indecies):
    batch_size = Xi.size(1)
    sam_size = X.size(1)

    dist_map = Xi.pow(2).sum(dim=0, keepdim=True).expand(sam_size, batch_size) + \
               X.pow(2).sum(dim=0, keepdim=True).expand(batch_size, sam_size).t()
    dist_map = dist_map.t()
    dist_map.addmm_(1, -2, Xi.t(), X)

    trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
    indecies = torch.from_numpy(np.array(indecies)).cuda()
    neg, _ = dist_map[trick!=indecies.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
    neg = torch.sqrt(neg.mean(dim=1))
    return neg

def ConstrainX(Xi):
    s_T = torch.sum(Xi.mul(Xi), 0).unsqueeze(0)
    allone = torch.ones_like(s_T)
    err = allone - s_T
    contmp = err.mm(err.t()).squeeze()
    contmp = contmp.div(Xi.size(1))
    return contmp


def CountLoss(Xi_, Bi_, X, B,loss_weights, scale, indecies, batch_idx,sel_neg_indecies):
    show_view_ind = 0
    num_view = len(Xi_)
    Xloss = torch.zeros(num_view).cuda()
    Bloss = torch.zeros(num_view).cuda()
    Qloss = torch.zeros(num_view).cuda()
    Conloss = torch.zeros(num_view).cuda()
    loss = torch.zeros(num_view).cuda()
    xwei, bwei, qwei, lamda= loss_weights

    lossFunc = BCEWithLogitsLoss(reduction='mean')
    part_B = B[:, indecies]
    B01 = torch.where(part_B==-1, torch.zeros_like(part_B), torch.ones_like(part_B))

    for view_ind, Xi in enumerate(Xi_):
        qtmp = lossFunc(Bi_[view_ind], B01)

        contmp = ConstrainX(Xi)
        if sel_neg_indecies is not None:
            neg = PartXmulloss(Xi,X[:,sel_neg_indecies])
            neg_b = PartXmulloss(Bi_[view_ind],B[:,sel_neg_indecies])
        else:
            neg = CalXmulloss(Xi,X,indecies)
            neg_b = CalXmulloss(Bi_[view_ind],B,indecies)

        xdiff = torch.sqrt(torch.pow((X[:,indecies].data-Xi), 2).sum(dim=0))
        x = (-scale * xdiff).exp().sum()
        y = (-scale * neg).exp().sum()
        xtmp = y.log()-x.log()

        bdiff = torch.sqrt(torch.pow((part_B.data-Bi_[view_ind]), 2).sum(dim=0))
        xb = (-scale * bdiff).exp().sum()
        yb = (-scale * neg_b).exp().sum()
        btmp = yb.log()-xb.log()

        # print("---xtmp---")
        # print("x: {}".format(x.data))
        # print("y: {}".format(y.data))
        # print("xtmp {:.3f}".format(xtmp.data))
        # print("---btmp---")
        # print("xb: {}".format(xb.data))
        # print("yb: {}".format(yb.data))
        # print("btmp {:.3f}".format(btmp.data))
        
        Xloss[view_ind] = xtmp
        Bloss[view_ind] = btmp
        Qloss[view_ind] = qtmp
        Conloss[view_ind] = contmp

        loss[view_ind] = xwei*xtmp + bwei*btmp + qwei*qtmp + lamda*contmp
        loss_parts = [Xloss,Bloss,Qloss,Conloss]
    total_loss = torch.sum(loss)

    if batch_idx % 500 == 0:
        print(red("--- batch idx {} ---".format(batch_idx)))
        print("Xi[{}] range {:.2f}~{:.2f}".format(show_view_ind, torch.min(Xi_[show_view_ind]),
                                                  torch.max(Xi_[show_view_ind])))

        print("({})Xloss {:.3f}".format(xwei, torch.sum(Xloss)))
        print("({})Bloss {:.3f}".format(bwei, torch.sum(Bloss)))
        print("({})Qloss {:.3f}".format(qwei, torch.sum(Qloss)))
        print("({})Conloss {:.3f}".format(lamda, torch.sum(Conloss)))

        print("xtmp: {:.3f} ; btmp: {:.3f}".format(xtmp.data,btmp.data))

    return total_loss,loss_parts


def UpdateAlpha(loss_parts,loss_weights,gamma):
    # xl,bl,ql,cl = loss_parts
    xl,bl,ql,cl = [i.data.cpu() for i in loss_parts]
    assert xl.size(0) == bl.size(0) == ql.size(0) == cl.size(0)
    xwei,bwei,qwei,lamda = loss_weights

    L = torch.zeros_like(xl)
    num_view = xl.size(0)
    for view_ind in range(num_view):
        Li = xwei * xl[view_ind] + bwei * bl[view_ind] + qwei * ql[view_ind] + lamda * cl[view_ind]
        if gamma > 1:
            Li = torch.pow(Li,(1.0/(1.0-gamma)))
        L[view_ind] = Li

    cur_alpha = L / torch.sum(L)
    return cur_alpha
