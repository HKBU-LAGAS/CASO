import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class CASO(nn.Module):
    def __init__(self, args, dataset):
        super(CASO, self).__init__()
        # self.dataset = dataset
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.latent_dim = args.latent_dim
        self.l2_reg = args.l2_reg
        self.pool_beta = args.pool_beta
        self.mat_norm = args.mat_norm
        self.kl_beta = args.kl_beta
        self.edge_bias = args.edge_bias
        self.Z_beta = args.Z_beta
        # option
        self.modularity_opt=args.modularity_opt
        self.Z2_type = args.Z2_type
        self.Z_begin = args.Z_begin
        self.init_type = args.init_type
        self.pool_type = args.pool_type
        self.batch_size = args.batch_size
        self.MLP_opt = args.MLP_opt
        self.ZZ_HSIC_opt = args.ZZ_HSIC_opt
        self.ZR_HSIC_opt = args.ZR_HSIC_opt
        # new parameter
        self.social_edge = dataset.social_edge
        self.community_edge = dataset.community_edge
        self.Z_alpha = args.Z_alpha
        self.Z_layer = args.Z_layer
        self.HSIC_layer = args.HSIC_layer
        self.HSIC_lambda = args.HSIC_lambda
        self.A = dataset.A
        self.AA_norm = dataset.AA_norm
        self.A_norm = dataset.A_norm
        self.A_row_norm = dataset.A_row_norm
        self.A_deg = dataset.A_deg.to_dense()
        self.A_deg_t = dataset.A_deg_t.to_dense()
        self.A_sqrt = dataset.A_sqrt
        self.D_log = dataset.D_log
        self.D = dataset.D
        self.B = dataset.B
        self.B_norm = dataset.B_norm
        self.B_row_norm = dataset.B_row_norm
        self.B_line_norm = dataset.B_line_norm
        self.B_deg = dataset.B_deg.to_dense()
        self.B_deg_t = dataset.B_deg_t.to_dense()
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self._init_weights()

    def _init_weights(self):
        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)

        # init strategy
        if self.init_type == 'no':
            nn.init.normal_(self.user_embeddings.weight, std=0.1)
            nn.init.normal_(self.item_embeddings.weight, std=0.1)
        elif self.init_type == 'orthogonal':
            nn.init.orthogonal_(self.user_embeddings.weight, gain=1.0)
            nn.init.orthogonal_(self.item_embeddings.weight, gain=1.0)
        elif self.init_type == 'kaiming':
            nn.init.kaiming_normal_(self.user_embeddings.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_normal_(self.item_embeddings.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.init_type == 'xa_norm':
            nn.init.xavier_normal_(self.user_embeddings.weight)
            nn.init.xavier_normal_(self.item_embeddings.weight)
        else:
            raise NotImplementedError

        return None

    def deg_norm(self, mat):
        r = torch.sum(mat, dim=1,keepdim=True)
        # d = torch.where(d == 0, torch.tensor(1e-9), d)
        dmat = torch.mm(mat.t(),r)
        dmat[dmat==0.] = 1.
        d = dmat.pow(-1).flatten()
        # d = torch.where(torch.isinf(d), torch.tensor(0.0), d)
        diag = torch.diag(d)
        mat_norm = torch.mm(mat,diag)
        return mat_norm
    def F_norm(self, R):
        s = torch.sqrt(torch.sum(R*R))
        return R / s
    def HSIC(self, Z, R):
        lamb = self.HSIC_lambda
        denoise_Z = Z
        denoise_R = R
        for _ in range(self.HSIC_layer):
            # matrix norm
            if self.mat_norm == 'l2':
                denoise_Z_norm = F.normalize(denoise_Z, p=2, dim=1)
                denoise_R_norm = F.normalize(denoise_R, p=2, dim=1)
            elif self.mat_norm == 'deg':  # 'deg'
                denoise_Z_norm = self.deg_norm(denoise_Z)
                denoise_R_norm = self.deg_norm(denoise_R)
            elif self.mat_norm == 'F_norm':
                denoise_Z_norm = self.F_norm(denoise_Z)
                denoise_R_norm = self.F_norm(denoise_R)
            else:  # no
                denoise_Z_norm = denoise_Z
                denoise_R_norm = denoise_R
            RRZ = torch.mm(denoise_R_norm, torch.mm(denoise_R_norm.t(), denoise_Z))
            if self.mat_norm == 'no':
                RRZ = F.normalize(RRZ, p=2, dim=1)
            denoise_Z = denoise_Z - lamb * RRZ

            ZZR = torch.mm(denoise_Z_norm, torch.mm(denoise_Z_norm.t(), denoise_R))
            if self.mat_norm == 'no':
                ZZR = F.normalize(ZZR, p=2, dim=1)
            denoise_R = denoise_R - lamb * ZZR
        return denoise_Z, denoise_R
    def forward(self):
        # get Z
        U = self.user_embeddings.weight
        start_U = U
        if self.modularity_opt == 'B':
            start_U = torch.mm(self.B, torch.mm(self.B_line_norm.transpose(0,1), U))
        Z = start_U
        Z_para = self.Z_alpha/(1-self.Z_alpha)
        reci_social = 1.0 / self.social_edge
        last_Z = start_U
        for i in range(self.Z_layer):
            tmp_AU = torch.spmm(self.A_norm, last_Z)
            tmp_dU = self.edge_bias * reci_social * torch.mm(self.A_deg, torch.mm(self.A_deg_t, last_Z))
            last_Z = Z_para * (tmp_AU - tmp_dU)
            Z = Z + last_Z

        Z1 = Z
        Z1_norm = F.normalize(Z1, p=2, dim=1)
        # get R
        B_bias = self.edge_bias
        reci_comm = 1.0 / self.community_edge
        tmp_BC = torch.spmm(self.B_norm.transpose(0,1), U)
        tmp_dC = B_bias * reci_comm * torch.mm(self.B_deg_t.T, torch.mm(self.B_deg.T, U))
        tmp_R = tmp_BC - tmp_dC
        tmp_BC = torch.spmm(self.B_norm, tmp_R)
        tmp_dC = B_bias * reci_comm * torch.mm(self.B_deg, torch.mm(self.B_deg_t, tmp_R))
        R = tmp_BC - tmp_dC
        R_norm = F.normalize(R, p=2, dim=1)
        # local social
        Z2 = U
        last_Z = U
        if self.Z2_type == 'AA':
            for i in range(2):
                last_Z = torch.spmm(self.A_norm, last_Z)
            Z2 = last_Z
        elif self.Z2_type =='CN':
            last_Z = torch.spmm(self.A, U)
            last_Z = torch.spmm(self.A_row_norm, last_Z)
            Z2 = last_Z
        elif self.Z2_type == 'AAI':
            last_Z = torch.spmm(self.A, U)
            last_Z = torch.spmm(self.D_log, last_Z)
            last_Z = torch.spmm(self.A_row_norm, last_Z)
            Z2 = last_Z
        elif self.Z2_type == 'RAI':
            last_Z = torch.spmm(self.A_row_norm, U)
            last_Z = torch.spmm(self.A_row_norm, last_Z)
            Z2 = last_Z
        elif self.Z2_type == 'SI':
            last_Z = torch.spmm(self.A_sqrt.T, U)
            last_Z = torch.spmm(self.A_sqrt, last_Z)
            last_Z = torch.spmm(self.D, last_Z)
            Z2 = last_Z
        elif self.Z2_type == 'LHNI':
            last_Z = torch.spmm(self.A_row_norm.T, U)
            last_Z = torch.spmm(self.A_row_norm, last_Z)
            last_Z = torch.spmm(self.D, last_Z)
            Z2 = last_Z
        elif self.Z2_type =='new_A':
            reci_social = 1.0 / self.social_edge
            for i in range(1):
                tmp_AU = torch.spmm(self.A_norm, last_Z)
                tmp_dU = self.edge_bias * reci_social * torch.mm(self.A_deg, torch.mm(self.A_deg_t, last_Z))
                last_Z = (tmp_AU - tmp_dU)
            Z2 = last_Z
        elif self.Z2_type =='simrank':
            # last_Z = torch.spmm(self.AA_norm, last_Z)
            last_Z = torch.spmm(self.A_row_norm.T, U)
            last_Z = torch.spmm(self.A_row_norm, last_Z)
            Z2 = last_Z
        elif self.Z2_type =='norm':
            # last_Z = torch.spmm(self.AA_norm, last_Z)
            last_Z = torch.spmm(self.A_norm, U)
            last_Z = torch.spmm(self.A_norm, last_Z)
            Z2 = last_Z
        elif self.Z2_type =='ADA':
            last_Z = torch.spmm(self.A_row_norm, U)
            last_Z = torch.spmm(self.A, last_Z)
            Z2 = last_Z
        elif self.Z2_type =='OCN':
            last_Z = torch.spmm(self.A, U)
            last_Z = torch.spmm(self.A, last_Z)
            Z2 = last_Z
        elif self.Z2_type == 'OAAI':
            last_Z = torch.spmm(self.A, U)
            last_Z = torch.spmm(self.D_log, last_Z)
            last_Z = torch.spmm(self.A, last_Z)
            Z2 = last_Z
        elif self.Z2_type == 'ORAI':
            last_Z = torch.spmm(self.A_row_norm, U)
            last_Z = torch.spmm(self.A, last_Z)
            Z2 = last_Z
        else : # AA*A
            last_Z = torch.spmm(self.AA_norm, last_Z)
            Z2 = last_Z
        Z2_norm = F.normalize(Z2, p=2, dim=1)
        if self.ZZ_HSIC_opt == 'yes':
            if self.half_HSIC_opt == 'yes':
                Z1_denoise, Z2_denoise = self.HSIC(Z1, Z2)
            else :
                Z1_denoise, Z2_denoise = self.HSIC(Z1_norm, Z2_norm)
        else :
            Z1_denoise, Z2_denoise = Z1_norm, Z2_norm

        Z = self.Z_beta * Z1_denoise + (1 - self.Z_beta) * Z2_denoise

        self.Z1_denoise = Z1_denoise
        self.Z2_denoise = Z2_denoise
        Z_norm = F.normalize(Z, p=2, dim=1)

        # HSIC Z,R
        if self.ZR_HSIC_opt == 'yes':
            denoise_Z, denoise_R = self.HSIC(Z_norm, R_norm)
        else :
            denoise_Z, denoise_R = Z_norm, R_norm

        user_emb = self.pool_beta * denoise_Z + (1 - self.pool_beta) * denoise_R
        item_emb = self.item_embeddings.weight

        return user_emb, item_emb


    def getEmbedding(self, users, pos_items, neg_items):
        users_emb = self.user_emb[users]
        pos_emb = self.item_emb[pos_items]
        neg_emb = self.item_emb[neg_items]
        return users_emb, pos_emb, neg_emb


    def bpr_loss(self, users, pos_items, neg_items):
        (users_emb, pos_emb, neg_emb) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = 1/2 * (users_emb.norm(2).pow(2) +
                    pos_emb.norm(2).pow(2) +
                    neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return auc, bpr_loss, reg_loss*self.l2_reg

    def kl_loss(self, users, items, neg_items):
        user_emb_extend = self.user_emb[users]
        item_emb_extend = self.item_emb[items]
        gamma = 1.0
        # t-distribution
        q_mol = torch.pow(1.0 + torch.sum(torch.pow(user_emb_extend - item_emb_extend, 2), 1) / gamma, -(gamma + 1.0) / 2.0)
        q_neg = torch.pow(1.0 + torch.sum(torch.pow(user_emb_extend - self.item_emb[neg_items], 2), 1) / gamma, -(gamma + 1.0) / 2.0)

        unique_users, counts = torch.unique_consecutive(users, return_counts=True)

        q_deo = q_mol + q_neg
        q = (q_mol / q_deo)

        p = 1.0 / counts.repeat_interleave(counts)

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')
        return the_kl_loss

    def calculate_all_loss(self, users, pos_items, neg_items):
        self.user_emb, self.item_emb = self.forward()
        auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)
        kl_loss = self.kl_beta * self.kl_loss(users, pos_items, neg_items)
        loss = bpr_loss + reg_loss + kl_loss

        return auc, bpr_loss, reg_loss, kl_loss, loss
