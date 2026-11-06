import os
import re
import glob
#import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json
import json
from PIL import Image
import torch
from typing import Dict, List, Optional, Tuple, Union


# ─── Helpers ────────────────────────────────────────────────────────────────── ##chiranjeev - 5 june
from transformers import AutoProcessor, AutoTokenizer ##chiranjeev - 5 june
import transformers ##chiranjeev - 5 june
print("[INFO] Dataloader transformers:",transformers.__file__) ##chiranjeev - 5 june


def load_local_frame(path: str):
    image = Image.open(path).convert("RGB")
    MIN_RES = 28
    w, h = image.size
    if w < MIN_RES or h < MIN_RES:
        new_w = max(w, MIN_RES)
        new_h = max(h, MIN_RES)
        # print(f"[WARNING] resizing tiny image: {(w,h)} -> {(new_w,new_h)}")
        image = image.resize((new_w, new_h), Image.BICUBIC)
    # print("[INFO] Loaded image:", path, "size:", image.size)
    return image

def find_assistant_response_start(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Find the token position where the assistant's response begins.
    Looks for the last occurrence of '<|im_start|>assistant\n' pattern.
    The response tokens start right after this.
    """
    # Encode the assistant header
    # In Qwen2.5 chat format: <|im_start|>assistant\n
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # Find all <|im_start|> positions
    positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

    if len(positions) == 0:
        return 0

    # The last <|im_start|> is the assistant turn
    last_im_start = positions[-1].item()

    # The assistant header is: <|im_start|> assistant \n
    # We need to find where the actual content starts after this header
    # Encode "assistant\n" to know how many tokens to skip
    assistant_header_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    response_start = last_im_start + 1 + len(assistant_header_ids)

    return min(response_start, len(input_ids))


class PRCC(object):
    """ PRCC

    Reference:
        Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. TPAMI, 2019.

    URL: https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view
    """
    dataset_dir = 'PRCC/prcc'
    def __init__(self, root='/home/aac/chiranjeev/PRCC_Differ/DIFFER/data',caption_dir='/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/prcc',caption_model='EVA02-CLIP-L-14',load_sum_ft=False, **kwargs):
        caption_dir='/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/prcc'  ##chiranjeev
        # caption_model = 'EVA02-CLIP-L-14' # 'EVA02-CLIP-bigE-14'  ##chiranjeev
        root = '/home/aac/chiranjeev/PRCC_Differ/DIFFER/data' ##chiranjeev
        self.root = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # print("dataset_dir:",self.dataset_dir)

        # '''
        self.max_seq_len = 512
        self.system_message = "You are a helpful assistant."
        
        min_pixels = 56 * 5
        
        max_pixels = 14 * 14 * 50 * 80
        model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        # ── Processor (handles images + text together) ─────────────────── ##chiranjeev - 5 june
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.tokenizer = self.processor.tokenizer ##chiranjeev - 5 june

        # Add <second_ans> special token
        new_tokens = ["<second_ans>"]
        existing = self.tokenizer.special_tokens_map.get("additional_special_tokens", []) ##chiranjeev - 5 june
        updated = list(set(existing + new_tokens))
        self.tokenizer.add_special_tokens({"additional_special_tokens": updated})
        self.second_ans_id = self.tokenizer.convert_tokens_to_ids("<second_ans>")
        
        self.IGNORE_INDEX =  -100
        
        self.frame_to_row = {}  ##chiranjeev - 5 june
        with open("/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/prcc/vora_prcc_differ.json", "r") as f:  ##chiranjeev - 5 june
            for line in f:  ##chiranjeev - 5 june
                row = json.loads(line)  ##chiranjeev - 5 june
                self.frame_to_row[self.root+"/"+row["frames"][0]] = row  ##chiranjeev - 5 june
        
        # '''
        
        logger = logging.getLogger('DIFFER')
        self.logger=logger
        
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        self.val_dir = osp.join(self.dataset_dir, 'rgb/val')
        self.test_dir = osp.join(self.dataset_dir, 'rgb/test')
        self._check_before_run()
        
        self.load_sum_ft=load_sum_ft
        self.bio_index=list(map(int,kwargs['bio_index']))
        self.non_bio_index=list(map(int,kwargs['nonbio_index']))
        
       
        self.caption_dir = os.path.join(caption_dir,caption_model)

        self.ft_name='ft_'+caption_model
            
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes,num_camera = \
            self._process_dir_train(self.train_dir)
        # val, num_val_pids, num_val_imgs, num_val_clothes, _,_ = \
        #     self._process_dir_train(self.val_dir)
        num_val_pids, num_val_imgs, num_val_clothes = 0, 0, 0 ##chiranjeev - 23 july

        query_same, query_diff, gallery, num_test_pids, \
            num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs, \
            num_test_clothes, gallery_idx = self._process_dir_test(self.test_dir)

        num_total_pids = num_train_pids + num_test_pids
        num_test_imgs = num_query_imgs_same + num_query_imgs_diff + num_gallery_imgs
        num_total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        
        logger.info("=> PRCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset      | # ids | # images | # clothes")
        logger.info("  --------------------------------------------")
        logger.info("  train       | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  val         | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_clothes))
        logger.info("  test        | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query(same) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_same))
        logger.info("  query(diff) | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs_diff))
        logger.info("  gallery     | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  --------------------------------------------")
        logger.info("  total       | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  --------------------------------------------")
        

        self.train = train
        #self.val = val
        self.query_same = query_same
        self.query_diff = query_diff
        self.num_gallery_imgs=num_gallery_imgs
        self.num_query_imgs_diff = num_query_imgs_diff
        self.num_query_imgs_same = num_query_imgs_same
        self.gallery = gallery


        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes
        self.gallery_idx = gallery_idx
        self.num_camera=num_camera
        
        self.num_test_pids = num_test_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    # ''' 
    ##chiranjeev - 5 june
    def _build_messages(self, item: dict, probe_pil: Image.Image) -> list:
        """
        Build Qwen2.5-VL chat messages with the image in the user turn.

        Returns the messages list that processor.apply_chat_template expects.
        """
        human_question = item["conversations"][0]["value"]
        fine_answer_raw = item["conversations"][1]["value"]
        facial_answer = item["conversations_facial"][1]["value"]

        # Remove <image> and leading \n from fine answer
        fine_answer_clean = fine_answer_raw
        if fine_answer_clean.startswith("<image>\n"):
            fine_answer_clean = fine_answer_clean[len("<image>\n"):]
        elif fine_answer_clean.startswith("<image>"):
            fine_answer_clean = fine_answer_clean[len("<image>"):]

        # Merged answer: facial + <|im_end|> + <second_ans> + \n + fine
        merged_answer = (
            f"{facial_answer}<|im_end|><second_ans>\n{fine_answer_clean}"
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": probe_pil},  # image goes here
                    {"type": "text", "text": human_question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": merged_answer}],
            },
        ]
        return messages

    
    def _build_labels(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build labels (fine) and labels_facial from input_ids.

        1. Mask everything before the assistant response start → base label
        2. Split at <second_ans>:
           - labels_facial: keep only BEFORE <second_ans>
           - labels (fine): keep only FROM <second_ans> onward
        """
        label = input_ids.clone()

        # Step 1: Mask the prompt (everything before assistant response)
        response_start = find_assistant_response_start(input_ids, self.tokenizer)
        label[:response_start] = self.IGNORE_INDEX

        # Step 2: Split at <second_ans>
        label_facial = label.clone()
        ans_positions = (input_ids == self.second_ans_id).nonzero(as_tuple=True)[0]

        if ans_positions.numel() > 0:
            ans_pos = ans_positions[0].item()
            label_facial[ans_pos:] = self.IGNORE_INDEX   # facial: only before <second_ans>
            label[:ans_pos] = self.IGNORE_INDEX           # fine: only from <second_ans> onward

        return label, label_facial
        
    # '''
    
    def _process_dir_train(self, dir_path):
        if self.load_sum_ft:
            self.logger.info("loading summery feature")
            sum_id_file=os.path.join(os.path.dirname(self.caption_dir),'train_caption_summary_biometric.json')
            with open(sum_id_file,'r') as f:
                sum_id_info=json.load(f)
            
            sum_cloth_file=os.path.join(os.path.dirname(self.caption_dir),'train_caption_summary_clothes.json') ##chiranjeev
            with open(sum_cloth_file,'r') as f: ##chiranjeev
                sum_cloth_info=json.load(f) ##chiranjeev
            
            # first_key, first_value = next(iter(sum_id_info.items()))
            # feat_dim=np.asarray(first_value[self.ft_name]).shape[1]
        
        caption_path=os.path.join(self.caption_dir, dir_path.split('/')[-1]+'.npz')#'train.npz') #,dir_path.split('/')[-1]+'.npz') ##chiranjeev
        # print("caption_path:",caption_path, "caption_dir:",self.caption_dir)  ##chiranjeev
        ##caption_path: /home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/prcc/EVA02-CLIP-bigE-14/train.npz caption_dir: /home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/prcc/EVA02-CLIP-bigE-14
        # print("dir_path:",dir_path)  ##chiranjeev
        ## /home/aac/chiranjeev/PRCC_Differ/DIFFER/data/PRCC/prcc/rgb/train
        
        all_caption = np.load(caption_path, allow_pickle=True)
        all_caption_features = all_caption['data']
        all_caption_files = all_caption['metadata']
        ftNums=int(all_caption_features.shape[0]/all_caption_files.shape[0])
        all_caption_files=list(all_caption_files)
        
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        camera_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                camera_container.add(cam)
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir)+osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        camera_container=sorted(camera_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}
        # print("pid2label:",pid2label) #276-111, 190-50 ##chiranjeev
        # print("clothes2label:",clothes2label) #276-222,190-100 ##chiranjeev


        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        num_camera=len(camera_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] # 'A' or 'B' or 'C'
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes=osp.basename(pdir)
                else:
                    clothes=osp.basename(osp.basename(pdir)+osp.basename(img_dir)[0])
                clothes_id = clothes2label[clothes]
                # print(all_caption_features.shape, all_caption_features[2364].shape) ##chiranjeev
                # print(all_caption_features[(2364*6):((2364+1)*6)].shape) ##chiranjeev
                caption_feature_index=all_caption_files.index(img_dir[len(dir_path)+1:][:-4])
                # print("img_dir[len(dir_path)+1:][:-4]:",img_dir[len(dir_path)+1:][:-4], self.bio_index, self.non_bio_index, ftNums, caption_feature_index) ##chiranjeev
                # quit() ##chiranjeev
                caption_feature_load=all_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
                caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
                
                biometric_caption_text = sum_id_info[str(pid)]["text"] ##chiranjeev
                non_biometric_caption_text = ' '.join(sum_cloth_info[str(clothes)]["text"]) ##chiranjeev
                # print("pid:",pid, "biometric_caption_text:",biometric_caption_text, "clothes_id:", clothes, "non_biometric_caption_text:",non_biometric_caption_text) ##chiranjeev
                # quit()   
                if self.load_sum_ft and 0 in self.bio_index:
                    id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                    caption_feature[self.bio_index.index(0)]=id_ft
                    
                # if self.load_sum_ft and 2 in (self.bio_index+self.non_bio_index):
                #     cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
                #     cloth_index=(self.bio_index+self.non_bio_index).index(2)
                #     caption_feature[cloth_index]=cloth_ft
                #     # if 2 in self.non_bio_index:
                #     #     cloth_index=self.non_bio_index.index(2)
                #     #     caption_feature[1+cloth_index]=cloth_ft
                        
                # if self.load_sum_ft:
                #     id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                #     cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
                #     caption_feature=np.concatenate((id_ft,cloth_ft))
                # else:
                #     caption_path=os.path.join(self.caption_dir,os.path.splitext(img_dir[len(self.dataset_dir)+5:])[0]+'.npy')
                #     #os.path.abspath(caption_path)
                #     if os.path.isfile(caption_path):
                #         caption_feature=np.load(caption_path)
                #     else:
                #         logging.info(f"fail to find file {caption_path}")
                #         caption_feature=np.zeros((2,512))
                
                # '''
                # print("img_dir:",img_dir)
                item = self.frame_to_row.get(img_dir)
                probe_path = os.path.join(self.root, item["frames"][0])
                probe_pil = load_local_frame(probe_path)
                normal_path = os.path.join(self.root, item["frames_normal"][0])
                # normal_pil = load_local_frame(normal_path)
                normal_cam_id = item["normal_cam_id"]
                # ── Build chat messages and process ──────────────────────────────
                messages = self._build_messages(item, probe_pil)
                # apply_chat_template produces the text with image placeholders
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # Processor tokenizes text AND processes the image
                inputs = self.processor(
                    text=[text],
                    images=[probe_pil],
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_seq_len,
                )
                del probe_pil
                
                input_ids = inputs["input_ids"].squeeze(0)             # [seq_len]
                attention_mask = inputs["attention_mask"].squeeze(0)    # [seq_len]
                pixel_values = inputs["pixel_values"]                  # [n_patches, 3, pH, pW]
                image_grid_thw = inputs["image_grid_thw"]              # [1, 3]
                
                # ── Build dual labels ────────────────────────────────────────────
                labels, labels_facial = self._build_labels(input_ids)

                # '''

                # data={
                #     "image_path": img_dir,
                #     "pid": label,
                #     "camid": camid,
                #     "clothes_id": clothes_id,
                #     "caption_feature":caption_feature,
                #     }
                
                data={
                    # DIFFER inputs
                    "image_path": img_dir,
                    "pid": label,
                    "camid": camid,
                    "clothes_id": clothes_id,
                    "caption_feature":caption_feature,
                                
                    # ReID backbone inputs
                    "arcface_frame_normal_gal_path": normal_path,
                    "normal_cam_id": normal_cam_id,
                    
                    # Qwen2.5-VL inputs (processor handles image tokens)
                    "input_ids": input_ids,                              # [seq_len]
                    "attention_mask": attention_mask,                    # [seq_len]
                    "pixel_values": pixel_values,                        # [n_patches, 3, pH, pW]
                    "image_grid_thw": image_grid_thw, 

                    # Dual-head labels
                    "labels": labels,                                    # [seq_len]
                    "labels_facial": labels_facial,                      # [seq_len]
                }
                           
                dataset.append(data) 
                
                #dataset.append((img_dir, label, camid, clothes_id,caption_feature))
                pid2clothes[label, clothes_id] = 1            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes,num_camera

    def _process_dir_test(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []
        
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    pid_cls = pid2label[pid]
                    camid = cam2label[cam]
                    #caption_feature=np.zeros((2,1024))
                    data={
                        "image_path": img_dir,
                        "pid": pid_cls,
                        "camid": camid,
                        "clothes_id": 0,
                        }
            
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        data["clothes_id"]=clothes_id
                        gallery_dataset.append(data) 
                        #gallery_dataset.append((img_dir, pid, camid, clothes_id,caption_feature))
                    elif cam == 'B':
                        clothes_id = pid2label[pid] * 2
                        data["clothes_id"]=clothes_id
                        query_dataset_same_clothes.append(data) 
                        #query_dataset_same_clothes.append((img_dir, pid, camid, clothes_id,caption_feature))
                    else:
                        clothes_id = pid2label[pid] * 2 + 1
                        data["clothes_id"]=clothes_id
                        query_dataset_diff_clothes.append(data) 
                        #query_dataset_diff_clothes.append((img_dir, pid, camid, clothes_id,caption_feature))

        pid2imgidx = {}
       
        for idx, data in enumerate(gallery_dataset):
            if data["pid"] not in pid2imgidx:
                pid2imgidx[pid] = []
            pid2imgidx[pid].append(idx)

        # get 10 gallery index to perform single-shot test
        gallery_idx = {}
        random.seed(3)
        for idx in range(0, 10):
            gallery_idx[idx] = []
            for pid in pid2imgidx:
                gallery_idx[idx].append(random.choice(pid2imgidx[pid]))
                 
        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
               num_pids, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery, \
               num_clothes, gallery_idx
