from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os
from os.path import join as pjoin
import torch
import torch.utils.data as data
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from mlp.data.abstract_dataset import AbstractDataset
from mlp.utils import io_utils, net_utils

np.set_printoptions(precision=4)


def create_loaders(split, data_config: DictConfig):
    dsets, L = {}, {}
    for di, dt in enumerate(split):
        shuffle = True if dt == "train" else False
        drop_last = True if dt == "train" else False
        dsets[dt] = instantiate(data_config, split=dt)
        L[dt] = data.DataLoader(
            dsets[dt],
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
            shuffle=shuffle,  # shuffle
            collate_fn=dsets[dt].collate_fn,
            drop_last=drop_last  # drop_last
        )
    return dsets, L


class BabelDataset(AbstractDataset):
    def __init__(self, 
                 dataname: str,
                 nums_snippet: int,
                 datapath: str,
                 labelpath: str,
                 batch_size: int,
                 threshold: int,
                 training_checked: bool,
                 t2s_model,
                 split: str = "train",
                 extend: float = 0.0,
                 before_norm: bool = False,
                 in_memory: bool = False,
                 contain_extra: bool = False,
                 debug: bool = False,
                 **kwargs):
        super(BabelDataset, self).__init__()

        # get options
        self.S = nums_snippet
        self.split = split
        self.label_dir = labelpath
        self.feature_dir = datapath
        self.training_checked = training_checked
        self.in_memory = in_memory
        self.extend = extend
        self.batch_size = batch_size
        self.debug = debug
        self.before_norm = before_norm

        # get paths for proposals and captions
        paths, ts_ckd_path = self._get_data_path(split, nums_snippet, threshold, labelpath)

        # create labels (or load existing one)
        ann_path = os.path.join(self.label_dir, f"{dataname}.json")
        extra_ann_path = os.path.join(self.label_dir, f"{dataname}_extra.json")
        self.anns, self.qids, self.mpaths, self.checked_timestamps = self._load_annotation(
            ann_path, extra_ann_path, t2s_model, threshold, ts_ckd_path, contain_extra)
        if not self._exist_data(paths):
            self.generate_labels()
        del t2s_model

        # load features if use in_memory
        if self.in_memory:
            self.feats = {}
            for mid in tqdm(self.mpaths, desc="In-Memory: mid_feat"):
                feature = np.load(f"{self.feature_dir}/{mid}.npy")
                self.feats[mid] = self.visual_feature_sampling(feature, self.S)

            self.s_pos, self.e_pos, self.att_mask, self.s_label, self.e_label, self.h_label = {}, {}, {}, {}, {}, {}
            suffix_key = 'ckd' if self.training_checked else 'gt'
            grd_info = io_utils.load_hdf5(self.paths["grounding_info"], False)
            for k in tqdm(self.qids, desc="In-Memory: grounding"):
                self.s_pos[k] = grd_info["start_pos/" + k][()]
                self.e_pos[k] = grd_info["end_pos/" + k][()]
                self.h_label[k] = grd_info[f"h_label_{suffix_key}/{k}"][()]
                self.s_label[k] = grd_info[f"s_label_{suffix_key}/{k}"][()]
                self.e_label[k] = grd_info[f"e_label_{suffix_key}/{k}"][()]
        
        if self.before_norm:
            self.mean = np.load(pjoin(labelpath, "Mean.npy"))
            self.std = np.load(pjoin(labelpath, "Std.npy"))
        
        self.num_instances = len(self.qids)
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        # get query id and corresponding video id
        qid = str(self.qids[idx])
        mid = self.anns[qid]["video_id"].split('_')[0]
        mpath = self.anns[qid]["path"]
        timestamp = self.anns[qid]["timestamps"]
        timestamps_ckd = self.checked_timestamps[qid]
        duration = self.anns[qid]["duration"]
        query = self.anns[qid]["query"]
        suffix_key = 'ckd' if self.training_checked else 'gt'

        # get grounding label
        if self.in_memory:
            start_label = self.s_label[qid]
            end_label = self.e_label[qid]
        else:
            grd_info = io_utils.load_hdf5(self.paths["grounding_info"], False)
            start_label = grd_info[f"s_label_{suffix_key}/{qid}"][()]
            end_label = grd_info[f"e_label_{suffix_key}/{qid}"][()]

        # get video features
        if self.in_memory:
            mid_feat = self.feats[mpath]
        else:
            mid_feat = self.visual_feature_sampling(
                np.load(f"{self.feature_dir}/{mpath}.npy"), self.S)
        
        if self.before_norm:
            "Z Normalization"
            motion = (motion - self.mean) / self.std
        # sampled nfeats
        nfeats = mid_feat.shape[0]

        mid_mask = np.ones((nfeats))

        # get highlight label
        if self.in_memory:
            h_label = self.h_label[qid]
        else:
            h_label = grd_info[f"h_label_{suffix_key}/{qid}"][()]

        instance = {
            "mids": mid,
            "qids": qid,
            "timestamps": timestamp,  # GT location [s, e] (second)
            "timestamps_ckd": timestamps_ckd,  # Checked location [[s1, e1],[s2,e2]] (second)
            "duration": duration,  # video span (second)
            "query": query,
            "nfeats": torch.FloatTensor([nfeats]),
            "motion_feats": torch.FloatTensor(mid_feat),  # [L_v,D_v]
            "motion_masks": torch.BoolTensor(mid_mask),  # [L_v,1]
            "grounding_h_labels": torch.IntTensor(h_label),
            "grounding_s_labels": torch.FloatTensor(start_label),
            "grounding_e_labels": torch.FloatTensor(end_label)
        }

        return instance

    def visual_feature_sampling(self, visual_feature, max_num_clips):
        num_clips = visual_feature.shape[0]
        if num_clips <= max_num_clips:
            return visual_feature
        idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
        idxs = np.round(idxs).astype(np.int32)
        idxs[idxs > num_clips - 1] = num_clips - 1
        new_visual_feature = []
        for i in range(max_num_clips):
            s_idx, e_idx = idxs[i], idxs[i + 1]
            if s_idx < e_idx:
                new_visual_feature.append(
                    np.mean(visual_feature[s_idx:e_idx], axis=0))
            else:
                new_visual_feature.append(visual_feature[s_idx])
        new_visual_feature = np.asarray(new_visual_feature)
        return new_visual_feature

    def collate_fn(self, data):
        seq_items = [
            "motion_feats", "motion_masks",
            "grounding_h_labels", "grounding_s_labels", "grounding_e_labels"
        ]
        tensor_items = [
            "nfeats",
        ]
        batch = {k: [d[k] for d in data] for k in data[0].keys()}

        # if len(data) == 1:
        #     for k, v in batch.items():
        #         if k in tensor_items:
        #             batch[k] = torch.cat(batch[k], 0)
        #         elif k in seq_items:
        #             batch[k] = torch.nn.utils.rnn.pad_sequence(
        #                 batch[k], batch_first=True)
        #         else:
        #             batch[k] = batch[k][0]

        # else:
        for k in tensor_items:
            batch[k] = torch.cat(batch[k], 0)
        for k in seq_items:
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                batch[k], batch_first=True)

        return batch

    def _get_data_path(self, split, S, threshold, base_dir):
        root_dir = os.path.join(base_dir, "preprocess")
        grounding_info_path = os.path.join(root_dir,
                                           "grounding_info", "{}_labels_S{}_sim{}.hdf5".format(split, S, threshold))
        query_info_path = os.path.join(root_dir,
                                           "query_info", "{}_timestamps_ckd_sim{}.json".format(split, threshold))
        

        io_utils.check_and_create_dir(os.path.join(root_dir, "grounding_info"))
        io_utils.check_and_create_dir(os.path.join(root_dir, "query_info"))

        self.paths = {
            "grounding_info": grounding_info_path
        }
        return self.paths, query_info_path

    def _load_annotation(self, ann_path, extra_path, t2s_model, threshold_selfsim, ts_ckd_path, extra=False):
        """ Load annotations
        Args:
            ann_paths: path for annotations; list or string
            extra_path: path for extra annotations; list or string
        Returns:
            new_anns: loaded and preprocessed annotations
        """
        path = [ann_path, extra_path] if extra else [ann_path]
        splits = [self.get_split_keyids(self.split), self.get_split_keyids(self.split+'_extra')] if extra else [self.get_split_keyids(self.split)]
        qid = 0
        new_anns = dict()
        preload = False
        if os.path.exists(ts_ckd_path):
            preload = True
            with open(ts_ckd_path, "r") as f:
                checked_timestamps = json.load(f)
        else:
            checked_timestamps = dict()
        mpaths = []
        for babel_file in path:
            with open(babel_file, "r") as f:
                json_data = json.load(f)
            valid_split = splits[1] if 'extra' in babel_file else splits[0]
            print(f'loading {babel_file}. It will take longer the first time...')
            for i in tqdm(json_data):
                if i not in valid_split:
                    continue
                item = json_data[i]
                if not os.path.exists(f"{self.feature_dir}/{item['path']}.npy"):
                    print(f"{str(i)}.npy not found, skipped")
                    continue
                if self.debug and qid>=self.batch_size:
                    break
                suffix = 'extra' if 'extra' in babel_file else ''
                labels = item['annotations']
                label_anns = []
                for label in labels:
                    query = label["text"]
                    # ignore "transition" action since since it does not convey
                    # consistent semantic information for describing motion
                    if query == "transition":
                        continue
                    label_anns.append({
                        "timestamps": [float(label['start']), float(label['end'])],
                        "query": query,
                        # "tokens": nltk.tokenize.word_tokenize(query.lower()),
                        "duration": float(item["duration"]),
                        "video_id": str(i) + suffix,
                        "path": item['path']
                    })
                    mpaths.append(item['path'])
                    
                if not preload and len(label_anns) > 0:
                    # Record timestamps with similar queries
                    sent_embs= t2s_model([x['query'] for x in label_anns])
                    # put the threshold value between -1 and 1
                    real_threshold_selfsim = 2 * threshold_selfsim - 1
                    selfsim = sent_embs @ sent_embs.T
                    selfsim_idxs = [torch.nonzero(row_mask).squeeze().tolist() for row_mask in selfsim > real_threshold_selfsim]
                    selfsim_idxs = [[ind] if isinstance(ind, int) else ind for ind in selfsim_idxs]
                for id, label_ann in enumerate(label_anns):
                    if not preload:
                        # Save all assigned timestamps
                        checked_timestamps[str(qid)] = [label_anns[j]["timestamps"] for j in selfsim_idxs[id]]
                    new_anns[str(qid)] = label_ann
                    qid += 1
        
        if not preload:
            ckd_ts_str = json.dumps(checked_timestamps)
            with open(ts_ckd_path, 'w') as f:
                f.write(ckd_ts_str)
                print(f"The checked timestamp is saved in {ts_ckd_path}")

        return new_anns, list(new_anns.keys()), list(set(mpaths)), checked_timestamps

    def generate_labels(self):
        """ Generate and save labels for temporal language grouding
            1)grounding_labels (.h5): qid -> label
        """

        """ Grounding information """
        if not os.path.exists(self.paths["grounding_info"]):
            grd_dataset = io_utils.open_hdf5(self.paths["grounding_info"], "w")
            start_pos = grd_dataset.create_group("start_pos")
            end_pos = grd_dataset.create_group("end_pos")
            start_index = grd_dataset.create_group("start_index")
            end_index = grd_dataset.create_group("end_index")
            h_label_gt = grd_dataset.create_group("h_label_gt")
            s_label_gt = grd_dataset.create_group("s_label_gt")
            e_label_gt = grd_dataset.create_group("e_label_gt")
            h_label_ckd = grd_dataset.create_group("h_label_ckd")
            s_label_ckd = grd_dataset.create_group("s_label_ckd")
            e_label_ckd = grd_dataset.create_group("e_label_ckd")

            for qid, ann in tqdm(self.anns.items(), desc="Gen. Grd. Labels"):
                # get starting/ending positions
                ts = ann["timestamps"]
                vid_d = ann["duration"]
                start = ts[0]
                end = ts[1]

                # get attention calibration mask
                m_path = ann["path"]
                # split by sep '_' is just for loading extra file
                nfeats = np.load(f"{self.feature_dir}/{m_path}.npy").shape[0]

                nfeats = min(nfeats, self.S)
                start_i, end_i, _ = net_utils.time_to_index(
                    start, end, nfeats, vid_d)

                # 0 for the background and 1 for the foreground
                hot_label_gt = np.zeros((nfeats))
                hot_label_gt[start_i:end_i + 1] = 1

                start_label_gt, end_label_gt = np.zeros((nfeats)), np.zeros((nfeats))
                start_label_gt[start_i], end_label_gt[end_i] = 1.0, 1.0

                # Assign multi-label data
                hot_label_ckd = np.zeros((nfeats))
                start_label_ckd, end_label_ckd = np.zeros((nfeats)), np.zeros((nfeats))
                s_idxs_ckd, e_idxs_ckd = [], []
                for timestamp in self.checked_timestamps[qid]:
                    s, e = timestamp[0], timestamp[1]
                    s_i, e_i, _ = net_utils.time_to_index(s, e, nfeats, vid_d)
                    hot_label_ckd[s_i:e_i+1] = 1
                    s_idxs_ckd.append(s_i)
                    e_idxs_ckd.append(e_i)
                
                # average label value
                start_label_ckd[s_idxs_ckd] = 1.0 / len(s_idxs_ckd)
                end_label_ckd[e_idxs_ckd] = 1.0 / len(s_idxs_ckd)

                _ = start_pos.create_dataset(qid, data=start, dtype="float")
                _ = end_pos.create_dataset(qid, data=end, dtype="float")
                _ = start_index.create_dataset(qid, data=start_i, dtype="int")
                _ = end_index.create_dataset(qid, data=end_i, dtype="int")
                _ = h_label_gt.create_dataset(qid, data=hot_label_gt, dtype="float")
                _ = s_label_gt.create_dataset(qid, data=start_label_gt, dtype="float")
                _ = e_label_gt.create_dataset(qid, data=end_label_gt, dtype="float")
                _ = h_label_ckd.create_dataset(qid, data=hot_label_ckd, dtype="float")
                _ = s_label_ckd.create_dataset(qid, data=start_label_ckd, dtype="float")
                _ = e_label_ckd.create_dataset(qid, data=end_label_ckd, dtype="float")

            # save the encoded proposal labels and motion ids
            grd_dataset.close()

    def get_extend_se_idx(self, nfeats, start_i, end_i):
        extend_len = round(self.extend * float(end_i - start_i + 1))
        if extend_len > 0:
            st_ = max(0, start_i - extend_len)
            et_ = min(end_i + extend_len, nfeats - 1)
            return st_, et_
        else:
            return start_i, end_i
