# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import random
import torch
import os
import pickle

from torch.utils.data import Dataset
from data_augmentation_time import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate

from utils import neg_sample, get_user_seqs, nCr
import copy


class Generate_tag():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name + "_1"
        self.save_path = save_path

    def generate(self):
        data_f = self.path + "/" + self.data_name + ".txt"
        print(data_f)
        train_dic = {}
        valid_dic = {}
        test_dic = {}
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(' ')
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-3]))
                valid_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train, [])
                train_dic[tag_train].append(train_temp)
                if tag_valid not in valid_dic:
                    valid_dic.setdefault(tag_valid, [])
                valid_dic[tag_valid].append(valid_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test, [])
                test_dic[tag_test].append(test_temp)

        total_dic = {"train": train_dic, "valid": valid_dic, "test": test_dic}
        print("Saving data to ", self.save_path)
        with open(self.save_path + "/" + self.data_name + "_t.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, time_seq, not_aug_users=None, test_neg_items=None, data_type='train',
                 similarity_model_type='offline', total_train_users=0):
        self.args = args
        self.time_seq = time_seq
        self.test_neg_items = test_neg_items
        self.max_len = args.max_seq_length
        self.not_aug_users = not_aug_users
        self.total_train_users = total_train_users

        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        if similarity_model_type == 'offline':
            self.similarity_model = args.offline_similarity_model
        elif similarity_model_type == 'online':
            self.similarity_model = args.online_similarity_model
        elif similarity_model_type == 'hybrid':
            self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'crop': Crop(args.crop_mode, args.crop_rate),
                              'mask': Mask(args.mask_mode, args.mask_rate),
                              'reorder': Reorder(args.reorder_mode, args.reorder_rate),
                              'substitute': Substitute(self.similarity_model, args.substitute_mode,
                                                       args.substitute_rate),
                              'insert': Insert(self.similarity_model, args.insert_rate, args.max_insert_num_per_pos),
                              'random': Random(args, self.similarity_model),
                              'combinatorial_enumerate': CombinatorialEnumerate(args, self.similarity_model)}

        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views

        self.data_type = data_type
        self.user_seq = user_seq
        self.model_warm_up_train_users = args.model_warm_up_epochs * len(user_seq)
        # create target item sets
        self.sem_tag = Generate_tag(self.args.data_dir, self.args.data_name, self.args.data_dir)
        self.train_tag = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t.pkl", "train")

        # self.true_user_id,_,_,_,_=get_user_seqs(args.train_data_file)

    def _one_pair_data_augmentation(self, input_ids, input_times, not_aug=False):
        """
        provides two positive samples given one sequence
        """
        augmented_seqs = []
        for i in range(2):
            if not_aug:
                assert self.args.not_aug_data_mode in ['zero', 'original']
                if self.args.not_aug_data_mode == 'zero':
                    # Return not augmented user data with 0
                    augmented_input_ids = [0] * self.max_len

                if self.args.not_aug_data_mode == 'original':
                    # Return not augmented user data with original data
                    pad_len = self.max_len - len(input_ids)
                    augmented_input_ids = [0] * pad_len + input_ids
                    augmented_input_ids = augmented_input_ids[-self.max_len:]
            else:
                augmented_input_ids = self.base_transform(input_ids, input_times)
                pad_len = self.max_len - len(augmented_input_ids)
                augmented_input_ids = [0] * pad_len + augmented_input_ids
                augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len
            cur_tensors = (torch.tensor(augmented_input_ids, dtype=torch.long))
            augmented_seqs.append(cur_tensors)

        return augmented_seqs

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        if (isinstance(copied_input_ids, list)):
            copied_input_ids = [0] * pad_len + copied_input_ids
        else:
            copied_input_ids = [0] * pad_len + copied_input_ids.tolist()
        copied_input_ids = copied_input_ids[-self.max_len:]
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        if type(target_pos) == tuple:
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len:]
            target_pos_2 = target_pos_2[-self.max_len:]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            if (isinstance(target_pos, list)):
                target_pos = [0] * pad_len + target_pos
            else:
                target_pos = [0] * pad_len + target_pos.tolist()
            target_neg = [0] * pad_len + target_neg
            target_pos = target_pos[-self.max_len:]
            target_neg = target_neg[-self.max_len:]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_pos) == tuple:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),   # user_id for training
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
            else:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for vailding
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(target_neg, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        global target_pos_
        user_id = index
        # t_user_id=self.true_user_id[index]
        items = self.user_seq[index]
        times = self.time_seq[index]
        input_times = times[:-2]

        self.total_train_users += 1
        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            temp = self.train_tag[items[-3]]
            flag = False
            for t_ in temp:
                if t_[1:] == items[:-3]:
                    continue
                else:
                    target_pos_ = t_[1:]
                    flag = True
            if not flag:
                target_pos_ = random.choice(temp)[1:]
            seq_label_signal = items[-2]  # no use
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            target_pos = (target_pos, target_pos_)
            # cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            # return (cur_rec_tensors)
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cf_tensors_list = []
            not_aug = False
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentation_pairs = nCr(self.n_views, 2)
            if self.total_train_users <= self.model_warm_up_train_users:
                total_augmentation_pairs = 0
            if (user_id in self.not_aug_users) and (self.total_train_users > self.model_warm_up_train_users):
                not_aug = True
            for i in range(total_augmentation_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids, input_times, not_aug))

            return cur_rec_tensors, cf_tensors_list
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)

# Dynamic Segmentation operations
def DS_default(i_file, o_file):
    """
    :param i_file: original data
    :param o_file: output data
    :return:
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    aug_d = {}
    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 3
        if len(item) > 53:
            while start < len(item) - 52:
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < 53:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start + 53])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")


# Dynamic Segmentation operations
def DS(i_file, o_file, max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :max_len: the max length of the sequence
    :return:
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 3
    # save
    max_keep_len = max_len + 2
    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        # item = d_.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 3
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - max_keep_len:
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start + max_save_len])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")

# Dynamic Segmentation operations
# def DS(i_file, o_file, max_len):
#     """
#     :param i_file: original data
#     :param o_file: output data
#     :max_len: the max length of the sequence
#     :return:
#     """
#     with open(i_file, "r+") as fr:
#         data = fr.readlines()
#     aug_d = []
#     # training, validation, and testing
#     max_save_len = max_len + 3
#     # save
#     max_keep_len = max_len + 2
#     for d_ in data:
#         # u_i, item = d_.split(' ', 1)
#         item = d_.split(' ')
#         # item = d_.split(' ')
#         item[-1] = str(eval(item[-1]))
#         # aug_d.setdefault(u_i, [])
#         start = 0
#         j = 3
#         if len(item) > max_save_len:
#             # training, validation, and testing
#             while start < len(item) - max_keep_len:
#                 j = start + 4
#                 while j < len(item):
#                     if start < 1 and j - start < max_save_len:
#                         aug_d.append(item[start:j])
#                         j += 1
#                     else:
#                         aug_d.append(item[start:start + max_save_len])
#                         break
#                 start += 1
#         else:
#             while j < len(item):
#                 aug_d.append(item[start:j + 1])
#                 j += 1
#     with open(o_file, "w+") as fw:
#             for i_ in aug_d:
#                 fw.write(' '.join(i_) + "\n")

class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


if __name__ == "__main__":
    # dynamic segmentation
    DS("../data/Beauty.txt", "../data/Beauty_1.txt", 10)
    # DS_default("../data/Beauty.txt", "../data/Beauty_1.txt")
    # generate target item
    g = Generate_tag("../data", "Beauty", "../data")
    # generate the dictionary
    data = g.get_data("../data/Beauty_1_t.pkl", "train")
    i = 0
    # Only one sequence in the data dictionary in the training phase has the target item ID
    for d_ in data:
        if len(data[d_]) < 2:
            i += 1
            print("less is : ", data[d_], d_)
    print(i)


# class data_augmentation():
#
#     def __init__(self, args, user_seq, time_seq, similarity_model_type='offline'):
#         self.args = args
#         self.user_seq = user_seq
#         self.time_seq = time_seq
#         self.max_len = args.max_seq_length
#         # it takes one sequence of items as input, and apply augmentation operation to get another sequence
#         if similarity_model_type == 'offline':
#             self.similarity_model = args.offline_similarity_model
#         elif similarity_model_type == 'online':
#             self.similarity_model = args.online_similarity_model
#         elif similarity_model_type == 'hybrid':
#             self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
#         print("Similarity Model Type:", similarity_model_type)
#         self.augmentations = {'crop': Crop(args.crop_mode, args.crop_rate),
#                               'mask': Mask(args.mask_mode, args.mask_rate),
#                               'reorder': Reorder(args.reorder_mode, args.reorder_rate),
#                               'substitute': Substitute(self.similarity_model, args.substitute_mode,
#                                                        args.substitute_rate),
#                               'insert': Insert(self.similarity_model, args.insert_rate, args.max_insert_num_per_pos),
#                               'random': Random(args, self.similarity_model),
#                               'combinatorial_enumerate': CombinatorialEnumerate(args, self.similarity_model)}
#
#         if self.args.base_augment_type not in self.augmentations:
#             raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
#         print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
#         self.base_transform = self.augmentations[self.args.base_augment_type]
#
#     def data_augmentation(self, user_seq, time_seq):
#         augmented_seqs = []
#         for index in range(len(user_seq)):
#             input_ids = user_seq[index]
#             input_times = time_seq[index]
#             augmented_input_ids = self.base_transform(input_ids, input_times)
#             pad_len = self.max_len - len(augmented_input_ids)
#             augmented_input_ids = [0] * pad_len + augmented_input_ids
#             augmented_input_ids = augmented_input_ids[-self.max_len:]
#
#             assert len(augmented_input_ids) == self.max_len
#             cur_tensors = (torch.tensor(augmented_input_ids, dtype=torch.long))
#             augmented_seqs.append(cur_tensors)
#         # 清除掉前面的0
#         # cleaned_seqs = [seq[torch.nonzero(seq, as_tuple=True)[0][0]:] for seq in augmented_seqs]
#         # cleaned_seqs = [augmented_seqs[augmented_seqs != 0]]
#         # 生成索引序列，假设从0开始编号
#         # indices = list(range(len(cleaned_seqs))
#
#         # 对列表中的每个张量进行处理
#         cleaned_seqs = [tensor[tensor != 0] for tensor in augmented_seqs]
#
#         print("augmented_seqs_length:", len(cleaned_seqs))
#
#         formatted_seqs = [' '.join(map(str, seq.numpy())) for seq in cleaned_seqs]
#
#         with open(self.args.aug_item_file, "w+") as fw:
#             for seq_str in formatted_seqs:
#                 fw.write(seq_str + '\n')
#         return augmented_seqs
