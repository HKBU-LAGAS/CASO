import os, pdb, sys
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import argparse
from shutil import copyfile
from time import time
from tqdm import tqdm
from set import *
from evaluate import *
from log import Logger
from CASO import CASO
from rec_dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='CASO Parameters')
    ### general parameters ###
    parser.add_argument('--cross_valid', type=str, default='no', help='use cross validation')
    parser.add_argument('--dataset', type=str, default='No input', help='dataset name')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
    parser.add_argument('--runid', type=str, default='0', help='current log id')
    parser.add_argument('--topk', type=int, default=5, help='Topk value for evaluation')
    parser.add_argument('--Topk', type=str, default='1,2,3,4,5', help='Topk value for evaluation')
    parser.add_argument('--device_id', type=str, default='0', help='device id')
    parser.add_argument('--vis_device', type=str, default='0', help='CUDA VISIBLE')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--early_stops', type=int, default=80, help='early stop patience')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negetiva samples for each [u,i] pair')
    parser.add_argument('--fold', type=int, default=0, help='K-fold')
    parser.add_argument('--num_user', type=int, default=-1, help='max uid')
    parser.add_argument('--num_item', type=int, default=-1, help='max iid')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--accu_step', type=int, default=1, help='accumulation steps')
    parser.add_argument('--auto_lr', type=str, default='step', help='scheduler')
    parser.add_argument('--extra_dir', type=str, default='', help='dir name')

    ### model parameters ###
    parser.add_argument('--init_type', type=str, default='orthogonal', help='embedding init:no/orthogonal')
    parser.add_argument('--kl_beta', type=float, default=0.01, help='kl_loss parameter')
    parser.add_argument('--l2_reg', type=float, default=0.001, help='?')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--pool_type', type=str, default='sum', help='cat/sum/max')
    parser.add_argument('--Z_alpha', type=float, default=0.33, help='need less than 0.33')
    parser.add_argument('--Z_layer', type=int, default=2, help='?')
    parser.add_argument('--Z2_type', type=str, default='AA', help='AA/AA*A')
    parser.add_argument('--edge_bias', type=float, default=1.0, help='bias of relations')
    parser.add_argument('--Z_beta', type=float, default=0.3, help='beta*Z1 || (1-beta)*Z2')
    parser.add_argument('--pool_beta', type=float, default=0.5, help='beta*Z || (1-beta)*R')
    parser.add_argument('--HSIC_layer', type=int, default=1, help='?')
    parser.add_argument('--HSIC_lambda', type=float, default=0.01, help='?')

    parser.add_argument('--mat_norm', type=str, default='F_norm', help='l2/deg/F_norm')
    parser.add_argument('--ZZ_HSIC_opt', type=str, default='no', help='HSIC between Z1 and Z2: yes/no')
    parser.add_argument('--ZR_HSIC_opt', type=str, default='yes', help='HSIC between Z and R: yes/no')
    parser.add_argument('--Z_begin', type=str, default='yes', help='add U: yes or no')
    parser.add_argument('--MLP_opt', type=str, default='no', help='MLP: yes or no')
    parser.add_argument('--self_loop', type=str, default='no', help='add self-loop: yes or no')
    parser.add_argument('--modularity_opt', type=str, default='no', help='no/B')
    return parser.parse_args()

def makir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval_test(model):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward()
    return user_emb.cpu().detach().numpy(), item_emb.cpu().detach().numpy()

def single_fold(mark, args, fold):
    record_path = 'saved/' + args.dataset + '/'
    makir_dir(record_path)
    log = Logger(record_path, 'CASO' + '-' + args.dataset + '-training' + '[' + str(fold + 1) + ']' + '.txt')
    all_log = Logger(record_path, 'CASO' + '-' + args.dataset + args.extra_dir +'-all_log' + '[' + str(fold + 1) + ']' + '.txt',
                     mode='a')
    result = Logger(record_path, 'CASO' + '-' + args.dataset + '[' + str(fold + 1) + ']' + '.log')
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    rec_data = Dataset(args)
    rec_model = CASO(args, rec_data)
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    rec_model.to(device)
    optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)
    #auto_lr
    if args.auto_lr == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) #exp
    elif args.auto_lr == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001) #cos
    elif args.auto_lr == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # no
    elif args.auto_lr == 'no':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1) #no
    else:
        raise NotImplementedError(f"auto_lr '{args.auto_lr}' is not yet supported.")
    # scheduler = get_warmup_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)

    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    topk = args.topk
    top = args.Topk.split(',')
    top = [int(num) for num in top]
    early_stop = args.early_stops
    accu_step = args.accu_step
    max_hr, max_recall, max_ndcg = {}, {}, {}
    val_max_hr, val_max_recall, val_max_ndcg = {}, {}, {}
    best_epoch = 0

    for epoch in tqdm(range(args.epochs), desc=set_color(f"Train:", 'pink'), colour='yellow',
                      dynamic_ncols=True, position=0):
        t1 = time()
        sum_auc, all_bpr_loss, all_reg_loss, all_kl_loss, all_total_loss, batch_num = 0, 0, 0, 0, 0, 0
        rec_model.train()
        #  batch数据
        loader = rec_data._batch_sampling(num_negative=args.num_neg)
        optimizer.zero_grad()
        for u, i, j in tqdm(loader, desc='All_batch'):
            u = torch.tensor(u).type(torch.long).to(device)  # [batch_size]
            i = torch.tensor(i).type(torch.long).to(device)  # [batch_size]
            j = torch.tensor(j).type(torch.long).to(device)  # [batch_size]
            auc, bpr_loss, reg_loss, kl_loss, total_loss = rec_model.calculate_all_loss(u, i, j)
            loss = total_loss / accu_step
            loss.backward()
            if (batch_num+1) % accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            sum_auc += auc.item()
            all_bpr_loss += bpr_loss.item()
            all_reg_loss += reg_loss.item()
            all_kl_loss += kl_loss.item()
            all_total_loss += total_loss.item()
            batch_num += 1
        if batch_num % accu_step > 0:
            optimizer.step()
            optimizer.zero_grad()

        if args.auto_lr != 'no':
            scheduler.step()

        mean_auc = sum_auc / batch_num
        mean_bpr_loss = all_bpr_loss / batch_num
        mean_reg_loss = all_reg_loss / batch_num
        mean_kl_loss = all_kl_loss / batch_num
        mean_total_loss = all_total_loss / batch_num
        log.write(set_color(
            'Epoch:{:d}, Type:{:s}[{:d}] Train_AUC:{:.4f}, Loss_rank:{:.8f}, Loss_reg:{:.8f}, Loss_kl:{:.8f}, Loss_sum:{:.8f}\n'
            .format(epoch,mark,fold, mean_auc, mean_bpr_loss, mean_reg_loss, mean_kl_loss, mean_total_loss), 'blue'))
        t2 = time()

        # ***************************  evaluation  *****************************#

        if epoch % 1 == 0:
            user_emb, item_emb = eval_test(rec_model)
            hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata, top, user_emb, item_emb,
                                                  rec_data.testdata.keys())
            val_hr, val_recall, val_ndcg = num_faiss_evaluate(rec_data.valdata, rec_data.traindata, top, user_emb,
                                                              item_emb,
                                                              rec_data.valdata.keys())
            for key in ndcg.keys():
                if key not in val_max_hr or val_hr[key] > val_max_hr[key]:
                    val_max_hr[key] = val_hr[key]
                    max_hr[key] = hr[key]
                    best_epoch = epoch
                if key not in val_max_recall or val_recall[key] > val_max_recall[key]:
                    val_max_recall[key] = val_recall[key]
                    max_recall[key] = recall[key]
                    best_epoch = epoch
                if key not in val_max_ndcg or val_ndcg[key] > val_max_ndcg[key]:
                    val_max_ndcg[key] = val_ndcg[key]
                    max_ndcg[key] = ndcg[key]
                    best_epoch = epoch

                log.write(set_color(
                    'Current Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.8f}, ndcg:{:.8f} val_recall:{:.8f}, val_ndcg:{:.8f}\n'.format(
                        epoch, key,
                        recall[key], ndcg[key], val_recall[key], val_ndcg[key]),
                    'green'))
                log.write(set_color(
                    'Best Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.8f}, ndcg:{:.8f} val_recall:{:.8f}, val_ndcg:{:.8f}\n'.format(
                        best_epoch, key,
                        max_recall[key], max_ndcg[key], val_max_recall[key], val_max_ndcg[key]),
                    'red'))

            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            # Early_stop
            if epoch > early_stop and epoch > best_epoch + early_stop:
                log.write('early stop: ' + str(epoch) + '\n')
                break
    log.close()
    all_log.close()
    result.close()
    return top, max_hr, max_recall, max_ndcg, best_epoch

def print_result(mark, out_dict, args, fold, top, max_hr, max_recall, max_ndcg, best_epoch):
    record_path = 'saved/' + args.dataset + '/'
    makir_dir(record_path)
    log = Logger(record_path, 'CASO' + '-' + args.dataset + '-training' + '[' + str(fold + 1) + ']' + '.txt', mode='a')
    all_log = Logger(record_path, 'CASO' + '-' + args.dataset + args.extra_dir + '-all_log' + '[' + str(fold + 1) + ']' + '.txt',
                     mode='a')
    result = Logger(record_path, 'CASO' + '-' + args.dataset + '[' + str(fold + 1) + ']' + '.log', mode='a')
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    print('Best Result')


    all_log_arg = ''
    for arg in vars(args):
        if arg in out_dict:
            all_log_arg += arg + '=' + str(getattr(args, arg)) + '\t'
    all_log_arg += 'Best epoch={:d}\t'.format(best_epoch)
    # all_log.write('dataset=' + args.dataset + ',' + 'pool_beta=' + str(args.pool_beta) + ',' + 'HSIC_lambda=' + str(args.HSIC_lambda) + ',' + 'pool_type=' + args.pool_type + ',' +'init_type=' + args.init_type + '\t')
    all_log.write(all_log_arg,0)
    for key in top:
        all_log.write('{:.8f}\t'.format(max_recall[key]), 0)
    for key in top:
        all_log.write('{:.8f}\t'.format(max_ndcg[key]), 0)
    all_log.write('\n', 0)

    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')
    log.write('Topk:\t[{:s}],\t HR\t Recall\t NDCG\t{:s}[{:d}]\tBest epoch:{:d}\n'.format(args.Topk,mark,fold,best_epoch))
    for key in top:
        log.write('Topk:{:3d},\t {:.8f}\t{:.8f}\t{:.8f}\n'.format(key, max_hr[key], max_recall[key], max_ndcg[key]))

    for arg in vars(args):
        result.write(arg + '=' + str(getattr(args, arg)) + '\n')
    result.write('Topk:\t[{:s}],\t HR\t Recall\t NDCG\t{:s}[{:d}]\tBest epoch:{:d}\n'.format(args.Topk,mark,fold,best_epoch))
    for key in top:
        result.write('Topk:{:3d},\t{:.8f}\t{:.8f}\n'.format(key, max_recall[key], max_ndcg[key]))

    log.close()
    all_log.close()
    result.close()
    print('END')
    # ***********************************  Best Result   ********************************#



if __name__ == '__main__':
    seed_everything(2024)
    args = parse_args()
    if args.dataset == 'BlogCatalog':
        args.num_user = 5196
        args.num_item = 6
    elif args.dataset == 'Flickr':
        args.num_user = 7575
        args.num_item = 9
    elif args.dataset == 'Deezer-HR':
        args.num_user = 54573
        args.num_item = 84
    elif args.dataset == 'Deezer-HU':
        args.num_user = 47538
        args.num_item = 84
    elif args.dataset == 'Deezer-RO':
        args.num_user = 41773
        args.num_item = 84
    elif args.dataset == 'LiveJournal':
        args.num_user = 3997962
        args.num_item = 664414
    elif args.dataset == 'Youtube':
        args.num_user = 1134890
        args.num_item = 5000
    elif args.dataset == 'DBLP':
        args.num_user = 317080
        args.num_item = 5000
    elif args.dataset == 'Douban-Book':
        args.num_user = 13024
        args.num_item = 2936
    elif args.dataset == 'Yelp':
        args.num_user = 16239
        args.num_item = 511
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not yet supported.")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_device

    args.data_path = 'dataset/' + args.dataset + '/'

    # record parameter
    out_dict = {}
    out_dict['dataset'] = 1
    out_dict['lr'] = 1
    out_dict['Z_alpha'] = 1
    out_dict['Z_layer'] = 1
    out_dict['edge_bias'] = 1
    out_dict['pool_beta'] = 1
    out_dict['HSIC_lambda'] = 1
    # out_dict['gcn'] = 1
    # out_dict['Z_begin'] = 1
    out_dict['mat_norm'] = 1
    # out_dict['att_opt'] = 1
    out_dict['Z2_type'] = 1
    out_dict['kl_beta'] = 1
    out_dict['self_loop'] = 1
    out_dict['ZZ_HSIC_opt'] = 1
    out_dict['ZR_HSIC_opt'] = 1
    out_dict['Z_beta'] = 1
    out_dict['auto_lr'] = 1
    out_dict['ss_rate'] = 1
    out_dict['init_type'] = 1

    # save_file(record_path)
    if args.cross_valid =='yes': # 5-fold
        record_path = 'saved/' + args.dataset + '/'
        makir_dir(record_path)
        log = Logger(record_path, 'CASO' + '-' + args.dataset + args.extra_dir + '-CV_result' + '.txt', mode='a')
        tot_max_hr, tot_max_recall, tot_max_ndcg = defaultdict(float), defaultdict(float), defaultdict(float)
        top = []
        max_epoch = 0
        for fold in range(5):
            top, max_hr, max_recall, max_ndcg, best_epoch = single_fold('cv',args,fold)
            print_result('cv',out_dict, args, fold, top, max_hr, max_recall, max_ndcg, best_epoch)
            max_epoch = max(max_epoch, best_epoch)
            for key in top:

                tot_max_hr[key] += max_hr[key]
                tot_max_recall[key] += max_recall[key]
                tot_max_ndcg[key] += max_ndcg[key]

        log_arg = ''
        for arg in vars(args):
            if arg in out_dict:
                log_arg += arg + '=' + str(getattr(args, arg)) + '\t'
        log_arg += 'max_epoch=' + str(max_epoch) + '\t'
        log_arg += '\t'
        # all_log.write('dataset=' + args.dataset + ',' + 'pool_beta=' + str(args.pool_beta) + ',' + 'HSIC_lambda=' + str(args.HSIC_lambda) + ',' + 'pool_type=' + args.pool_type + ',' +'init_type=' + args.init_type + '\t')
        log.write(log_arg,0)
        for key in top:
            log.write('{:.8f}\t'.format(tot_max_recall[key]/len(top)), 1)
        for key in top:
            log.write('{:.8f}\t'.format(tot_max_ndcg[key]/len(top)), 1)
        log.write('\n', 0)

    else :
        top, max_hr, max_recall, max_ndcg, best_epoch = single_fold('one', args, args.fold)
        print_result('one',out_dict, args, args.fold, top, max_hr, max_recall, max_ndcg, best_epoch)
