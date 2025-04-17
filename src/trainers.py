# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os

import numpy as np
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from datasets import DS, RecWithContrastiveLearningDataset
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs
from models import KMeans
from modules import NCELoss


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model

        self.batch_size = self.args.batch_size
        self.sim = self.args.sim

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=64,
            gpu_id=args.gpu_id,
            device=torch.device("cuda"),
        )
        self.clusters = [cluster]
        self.clusters_t = [self.clusters]

        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.cf_criterion = NCELoss(self.args.temperature, self.device)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    # def __refresh_training_dataset(self, item_embeddings):
    #     """
    #     use for updating item embedding
    #     """
    #     # training data
    #     train_user_seq, train_time_seq, _, _, _, not_aug_users = get_user_seqs(self.args, self.args.train_data_file,
    #                                                                            self.args.train_time_file)
    #
    #     self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
    #     # training data for node classification
    #     train_dataset = RecWithContrastiveLearningDataset(self.args, train_user_seq, train_time_seq, not_aug_users=not_aug_users,
    #                                                       data_type='train', similarity_model_type='hybrid',
    #                                                       total_train_users=self.args.model_warm_up_epochs * len(train_user_seq)+1)
    #     train_sampler = RandomSampler(train_dataset)
    #     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
    #     return train_dataloader
    #
    # def __refresh_cluster_dataset(self, item_embeddings):
    #     """
    #     use for updating item embedding
    #     """
    #     # training data
    #     train_user_seq, train_time_seq, _, _, _, not_aug_users = get_user_seqs(self.args, self.args.train_data_file,
    #                                                                            self.args.train_time_file)
    #
    #     self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
    #     # training data for node classification
    #     cluster_dataset = RecWithContrastiveLearningDataset(self.args, train_user_seq, train_time_seq, not_aug_users=not_aug_users,
    #                                                       data_type='train', similarity_model_type='hybrid',
    #                                                       total_train_users=self.args.model_warm_up_epochs * len(train_user_seq)+1)
    #     cluster_sampler = RandomSampler(cluster_dataset)
    #     cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=self.args.batch_size)
    #     return cluster_dataloader

    def train(self, epoch):
        # start to use online item similarity
        # if epoch > self.args.augmentation_warm_up_epochs:
        #     print("refresh dataset with updated item embedding")
        #     if not os.path.exists(self.args.train_data_file):
        #         DS(self.args.data_file, self.args.train_data_file, self.args.max_seq_length)
        #     if not os.path.exists(self.args.train_time_file):
        #         DS(self.args.time_file, self.args.train_time_file, self.args.max_seq_length)
        #
        #     self.train_dataloader = self.__refresh_training_dataset(self.model.item_embeddings)
        #     self.cluster_dataloader = self.__refresh_cluster_dataset(self.model.item_embeddings)
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)
        return loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label = label.view(1, -1)
        label = label.expand((2, label.shape[-1])).reshape(1, -1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask == 0

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask == 0] = float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embedding(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self, coarse_intents, target_item):
        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :],
                                                       self.args.temperature, coarse_intent_1.shape[0], self.sim,
                                                       target_item[:, -1])
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss

    def ficl_loss(self, sequences, clusters_t):
        output = sequences[0][:, -1, :]
        intent_n = output.view(-1, output.shape[-1])  # [BxH]
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v = clusters_t[0].query(intent_n)

        seq_to_v = seq_to_v.view(seq_to_v.shape[0], -1)  # [BxH]
        a, b = self.info_nce(output.view(output.shape[0], -1), seq_to_v, self.args.temperature, output.shape[0],
                             sim=self.sim, intent_id=intent_id)
        loss_n_0 = nn.CrossEntropyLoss()(a, b)

        output_s = sequences[1][:, -1, :]
        intent_n = output_s.view(-1, output_s.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v_1 = clusters_t[0].query(intent_n)  # [BxH]
        seq_to_v_1 = seq_to_v_1.view(seq_to_v_1.shape[0], -1)  # [BxH]
        a, b = self.info_nce(output_s.view(output_s.shape[0], -1), seq_to_v_1, self.args.temperature, output_s.shape[0],
                             sim=self.sim, intent_id=intent_id)
        loss_n_1 = nn.CrossEntropyLoss()(a, b)
        ficl_loss = loss_n_0 + loss_n_1

        return ficl_loss


class TiICSRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(TiICSRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )

    def _one_pair_contrastive_learning(self, inputs):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]                                                                                                                                                                                                                                           // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0],
                                    cl_output_slice[1])
        return cl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):
        str_code = "train" if train else "test"
        if train:
            if self.args.cl_mode in ['cf', 'f']:
                # ------ intentions clustering ----- #
                print("Preparing Clustering:")
                self.model.eval()
                # save N
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                # for i, rec_batch in rec_cf_data_iter:
                    """
                    rec_batch shape: key_name x batch_size x feature_dim
                    cl_batches shape: 
                        list of n_views x batch_size x feature_dim tensors
                    """
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, subsequence, _, _, _ = rec_batch
                    sequence_output_a = self.model(subsequence)  # [BxLxH]
                    sequence_output_b = sequence_output_a[:, -1, :]  # [BxH]
                    kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                kmeans_training_data_t = [kmeans_training_data]

                for i, clusters in tqdm(enumerate(self.clusters_t), total=len(self.clusters_t)):
                    for j, cluster in enumerate(clusters):
                        cluster.train(kmeans_training_data_t[i])
                        self.clusters_t[i][j] = cluster

                # clean memory
                del kmeans_training_data
                del kmeans_training_data_t
                import gc
                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
            # for i, rec_batch in rec_cf_data_iter:
                """             
                rec_batch shape: key_name x batch_size x feature_dim
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, subsequence_1, target_pos_1, target_neg, subsequence_2 = rec_batch

                # ---------- prediction task -----------------------#
                intent_output = self.model(subsequence_1)
                logits = self.predict_full(intent_output[:, -1, :])  # [Bx|I|]
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos_1[:, -1])

                # ---------- recommendation task ---------------#
                # sequence_output = self.model.transformer_encoder(subsequence_1)
                # rec_loss = self.cross_entropy(sequence_output, target_pos_1, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                # ---------- intent representation learning task ---------------#
                coarse_intent_1 = self.model(subsequence_1)
                coarse_intent_2 = self.model(subsequence_2)
                if self.args.cl_mode in ['c', 'cf']:
                    cicl_loss = self.cicl_loss([coarse_intent_1, coarse_intent_2], target_pos_1)
                else:
                    cicl_loss = 0.0
                if self.args.cl_mode in ['f', 'cf']:
                    ficl_loss = self.ficl_loss([coarse_intent_1, coarse_intent_2], self.clusters_t[0])
                else:
                    ficl_loss = 0.0
                icl_loss = self.args.lambda_0 * cicl_loss + self.args.beta_0 * ficl_loss

                # ---------- multi-task learning --------------------#
                joint_loss = self.args.rec_weight * rec_loss + icl_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if type(icl_loss) != float:
                    icl_losses += icl_loss.item()
                else:
                    icl_losses += icl_loss

                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "icl_avg_loss": "{:.4f}".format(icl_losses / len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, _, answers = batch
                    recommend_output = self.model(input_ids)  # [BxLxH]
                    recommend_output = recommend_output[:, -1, :]  # [BxH]
                    # recommendation results
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batc in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
