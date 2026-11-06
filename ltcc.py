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

# import sys
# sys.path.append("/home/aac/chiranjeev/VoRA/transformers")  # Adjust this path as needed
# print("Dataloder:", sys.path) 

class LTCC(object):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'LTCC/LTCC_ReID'
    # def __init__(self, root='data',caption_dir='',caption_model='EVA02-CLIP-bigE-14',load_sum_ft=False, **kwargs): 
    def __init__(self, root='/home/aac/chiranjeev/PRCC_Differ/DIFFER/data',caption_dir='/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/ltcc',caption_model='EVA02-CLIP-L-14',load_sum_ft=False, **kwargs):
        # caption_dir='/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/ltcc'  ##chiranjeev
        # caption_model = 'EVA02-CLIP-L-14'  ##chiranjeev
        
        root = '/home/aac/chiranjeev/PRCC_Differ/DIFFER/data' ##chiranjeev
        self.root = root
        
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
        with open("/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/ltcc/vora_ltcc_differ.json", "r") as f:  ##chiranjeev - 5 june "/home/aac/chiranjeev/VoRA/data_module/vora_ltcc_differ.json"
            for line in f:  ##chiranjeev - 5 june
                row = json.loads(line)  ##chiranjeev - 5 june
                self.frame_to_row[self.root+"/"+row["frames"][0]] = row  ##chiranjeev - 5 june
        
        # '''
        
        logger = logging.getLogger('DIFFER')
        self.logger=logger
        
        self.dataset_dir = osp.join(root, self.dataset_dir)
       
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self._check_before_run()
        self.load_sum_ft=load_sum_ft
        self.bio_index=list(map(int,kwargs['bio_index']))
        self.non_bio_index=list(map(int,kwargs['nonbio_index']))
        
        self.caption_dir =os.path.join(caption_dir,caption_model)
        self.ft_name='ft_'+caption_model
        
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes,num_camera = \
            self._process_dir_train(self.train_dir)
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
      
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_test_imgs = num_query_imgs + num_gallery_imgs 
        num_total_clothes = num_train_clothes + num_test_clothes

        
        logger.info("=> LTCC loaded")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # clothes")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
        logger.info("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
        logger.info("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
        logger.info("  ----------------------------------------")
        logger.info("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
        logger.info("  ----------------------------------------")
        #self.logger=logger

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        
        self.num_query_imgs = num_query_imgs
        self.num_gallery_imgs = num_gallery_imgs
        self.pid2clothes = pid2clothes
        self.num_camera=num_camera
        self.num_test_pids = num_test_pids
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

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
                
            # # load clothing summery feature if needed
            # sum_cloth_file=os.path.join(self.caption_sum_dir,'train_caption_summary_clothes.json')
            # with open(sum_cloth_file,'r') as f:
            #     sum_cloth_info=json.load(f)
            #first_key, first_value = next(iter(sum_id_info.items()))
            #feat_dim=np.asarray(first_value[self.ft_name]).shape[1]
            
        caption_path=dir_path.replace(self.dataset_dir,self.caption_dir)+'.npz'
        all_caption = np.load(caption_path, allow_pickle=True)
        all_caption_features = all_caption['data']
        all_caption_files = all_caption['metadata']
        ftNums=int(all_caption_features.shape[0]/all_caption_files.shape[0])
        all_caption_files=list(all_caption_files)
            
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        camera_container = set()
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
            camera_container.add(camid)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        camera_container=sorted(camera_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        num_camera=len(camera_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        
        
        # print("img_paths:", img_paths)
        # quit()
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0

            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            
            caption_feature_index=all_caption_files.index(osp.basename(img_path)[:-4])
            caption_feature_load=all_caption_features[caption_feature_index*ftNums:(caption_feature_index+1)*ftNums]
            caption_feature=caption_feature_load[self.bio_index+self.non_bio_index]
            
                
            if self.load_sum_ft and 0 in self.bio_index:
                id_ft=np.asarray(sum_id_info[str(pid)][self.ft_name])
                caption_feature[self.bio_index.index(0)]=id_ft
                
            # if self.load_sum_ft and 2 in (self.bio_index+self.non_bio_index):
            #     cloth_ft=np.asarray(sum_cloth_info[clothes][self.ft_name])
            #     cloth_index=(self.bio_index+self.non_bio_index).index(2)
            #     caption_feature[cloth_index]=cloth_ft
            #caption_feature[1]=cloth_ft
            
            # '''
            item = self.frame_to_row.get(img_path)
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
            #     "image_path": img_path,
            #     "pid": pid,
            #     "camid": camid,
            #     "clothes_id": clothes_id,
            #     "caption_feature":caption_feature,
            #     }
            
            data={
                # DIFFER inputs
                "image_path": img_path,
                "pid": pid,
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
            #dataset.append((img_path, pid, camid, clothes_id,caption_feature))
            pid2clothes[pid, clothes_id] = 1
            
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes,num_camera

    def _process_dir_test(self, query_path, gallery_path):
        
        query_img_paths = glob.glob(osp.join(query_path, '*.png'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.png'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()

        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)

        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
       
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
          
            
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
            }
             
         
           
            query_dataset.append(data) 
          

        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            
          
            data={
                "image_path": img_path,
                "pid": pid,
                "camid": camid,
                "clothes_id": clothes_id,
                }
            
            gallery_dataset.append(data) 
           
           
           
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
