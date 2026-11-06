from utils.logger import setup_logger
from data import build_dataloader

from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from model import build_model
import copy

#from utilss import  auto_resume_helper,load_checkpoint

# import sys
# sys.path.append("/home/aac/chiranjeev/VoRA/transformers")  # Adjust this path as needed

# print(sys.path) 

from transformers import Qwen2_5_VLForConditionalGeneration #, BitsAndBytesConfig
print("[INFO] disent_train Qwen2_5_VLForConditionalGeneration:", Qwen2_5_VLForConditionalGeneration.__module__)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import disent_dataloader
import torch.nn as nn


# ============================================================
# Build Qwen model and apply LoRA
# ============================================================

def apply_lora(qwen_model=None, total_layers=24, rank=32, lora_alpha=8, lora_dropout=0.05):
    """
    Apply LoRA to Qwen2.5-VL-7B for dual-head ReID training.

    Key principles:
      - rank 16-64 is sufficient for 7B models (32 is a good default)
      - lora_alpha = 2 * rank is standard (scaling = alpha/rank = 2.0)
      - Target ALL transformer layers (not just first 24)
      - Include vision merger for better cross-modal learning
    """
    
    TARGET_MODULES = [
        ## LLM attention
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        ## LLM MLP
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]

    lora_target_modules = []
    for name, module in qwen_model.named_modules():
        ##LLM transformer layers — ALL of them
        if "model.layers." in name:
            if any(target in name for target in TARGET_MODULES):
                lora_target_modules.append(name.split("model.")[1])

        if "visual.merger" in name and isinstance(module, torch.nn.Linear):
            lora_target_modules.append(name)
            
        ## Vision-language merger (projects vision features into LLM space)
        # if "visual.merger" in name and "mlp" in name:
        #     lora_target_modules.append(name.split("model.")[1])

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )
    qwen_model = get_peft_model(qwen_model, peft_config)
    
    return qwen_model


# def build_qwen_model(total_layers=24, rank=512, lora_alpha=8, lora_dropout=0.0):
def build_qwen_model(total_layers=24, rank=32, lora_alpha=8, lora_dropout=0.05):
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    bnb_config = None
    USE_QLORA = False #True
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map={"": current_device},
            use_safetensors=True,
            )
    # initialize from pretrained lm_head_facial with lm_head weights
    qwen_model.lm_head_facial.weight.data.copy_(qwen_model.lm_head.weight.data)

    if USE_QLORA:
        qwen_model = prepare_model_for_kbit_training(qwen_model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},)
    else:
        qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        qwen_model.enable_input_require_grads()    
    
    qwen_model = apply_lora(qwen_model=qwen_model, total_layers=total_layers, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout) 
        
    for name, param in qwen_model.named_parameters():
        if "lm_head" in name or "lm_head_facial" in name or "hidden_facial" in name or "hidden_fine" in name:
            param.requires_grad = True
    
    # print(f"\n[Trainable Parameters]:")
    # for name, param in qwen_model.named_parameters():
    #     if param.requires_grad:
    #         print(f"  {name}: {param.shape}")
    qwen_model.config.use_cache = False
    return qwen_model


class IDClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(IDClassifier, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        return self.fc(x)


# ============================================================
# Build ReID model
# ============================================================

textModelVersionInfo={'EVA02-CLIP':{
    512:'B-16',
    768:'L-14',
    1024: 'bigE-14' #'B-16', #,
},
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.random.initial_seed()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CC_ReID  Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--jobId", default=0, help="Job number to identify different runs", type=int)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--qwen_required", action='store_true', help="Whether to use Qwen model or not") ##chiranjeev - argument to specify whether to use qwen model or not.
    args = parser.parse_args()
    print('[args]',args)
    
    
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    if args.loss is not None:
        loss_list=args.loss.split(',')
        args.opts+=['MODEL.LOSS_TYPE',loss_list]
    cfg.merge_from_list(args.opts)
    
        
    # if ('clipBioReverse' not in cfg.MODEL.LOSS_TYPE or 'clipBio' not in cfg.MODEL.LOSS_TYPE) and not args.qwen_required: ##chiranjeev
    #     cfg.MODEL.LAST_LAYER='transformer'
    print('cfg.MODEL.LOSS_TYPE:',cfg.MODEL.LOSS_TYPE, 'args.qwen_required:', args.qwen_required)
    if 'clipBioReverse' not in cfg.MODEL.LOSS_TYPE or 'clipBio' not in cfg.MODEL.LOSS_TYPE:
        cfg.MODEL.LAST_LAYER='transformer'
        
    output_dir = cfg.OUTPUT_DIR
    if args.jobId==0:
        if os.path.isdir(output_dir):
            model_index=len([x for x in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,x)) and x.find('_')>=0])
        else:
            model_index=0
    else:
        model_index=args.jobId
    outputfolder=f"{model_index}[{cfg.DATA.DATASET}][{cfg.MODEL.NAME}]".replace('/','')
    if cfg.OUTPUT_NAME:
        outputfolder=outputfolder+f"[{cfg.OUTPUT_NAME}]"
    cfg.OUTPUT_DIR=os.path.join(cfg.OUTPUT_DIR,outputfolder)
    if args.local_rank==0:
        os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
    
    textModelVersion=textModelVersionInfo[cfg.DATA.TEXT_MODEL][cfg.MODEL.CLIP_DIM]
    cfg.DATA.TEXT_MODEL_VERSION=f'{cfg.DATA.TEXT_MODEL}-{textModelVersion}'
    print('cfg.DATA.TEXT_MODEL:',cfg.DATA.TEXT_MODEL, "cfg.MODEL.CLIP_DIM:",cfg.MODEL.CLIP_DIM, "cfg.DATA.TEXT_MODEL_VERSION:",cfg.DATA.TEXT_MODEL_VERSION)


    cfg.freeze()
    
    #torch.use_deterministic_algorithms(True)
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    
    logger = setup_logger("DIFFER", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))

    if args.local_rank==0:
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                #logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
        with open(os.path.join(cfg.OUTPUT_DIR,'config.yml'), "w") as f:
            f.write(cfg.dump())

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
   
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.DATA.DATASET == 'prcc':
        if not args.qwen_required:
            trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(cfg)  # prcc_test
            tokenizer_dataset = None
        else:
            trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(cfg)  # prcc_test
            data_paths = [{
                "image_folder": f"{cfg.DATA.ROOT}/",
                "anno_path": "/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/prcc/vora_prcc_differ.json", #"/home/aac/chiranjeev/VoRA/data_module/vora_prcc_differ.json",
            }]
            # trainloader, tokenizer_dataset = disent_dataloader.build_train_dataloader(data_paths=data_paths, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", batch_size=cfg.DATA.BATCH_SIZE, max_seq_len=512, num_workers=cfg.DATA.NUM_WORKERS, training=True, distributed=cfg.MODEL.DIST_TRAIN)
            from transformers import AutoProcessor, AutoTokenizer ##chiranjeev - 5 june
            import transformers ##chiranjeev - 5 june
            min_pixels = 56 * 56
            max_pixels = 14 * 14 * 50 * 80
            model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            tokenizer = processor.tokenizer ##chiranjeev - 5 june
            tokenizer = processor.tokenizer ##chiranjeev - 5 june
            new_tokens = ["<second_ans>"]
            existing = tokenizer.special_tokens_map.get("additional_special_tokens", []) ##chiranjeev - 5 june
            updated = list(set(existing + new_tokens))
            tokenizer.add_special_tokens({"additional_special_tokens": updated})
            tokenizer_dataset = tokenizer
            
    else:
        if not args.qwen_required:
            trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader,trainOriginalLoader = build_dataloader(cfg)
            tokenizer_dataset = None
        else:
            trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader,trainOriginalLoader = build_dataloader(cfg)
            if cfg.DATA.DATASET == 'ltcc':
                data_paths = [{
                    "image_folder": f"{cfg.DATA.ROOT}/",
                    "anno_path": "/home/aac/chiranjeev/PRCC_Differ/TextCaptionDirectory/ltcc/vora_ltcc_differ.json", #"/home/aac/chiranjeev/VoRA/data_module/vora_ltcc_differ.json",
                }]
            elif cfg.DATA.DATASET == 'celeb_light':
                data_paths = [{
                    "image_folder": f"{cfg.DATA.ROOT}/",
                    "anno_path": "/home/aac/chiranjeev/VoRA/data_module/vora_Celeb_light_differ.json",
                }]
            # # trainloader, tokenizer_dataset = disent_dataloader.build_train_dataloader(data_paths=data_paths, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", batch_size=cfg.DATA.BATCH_SIZE, max_seq_len=512, num_workers=cfg.DATA.NUM_WORKERS, training=True)
            # trainloader, tokenizer_dataset = disent_dataloader.build_train_dataloader(data_paths=data_paths, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", batch_size=cfg.DATA.BATCH_SIZE, max_seq_len=512, num_workers=cfg.DATA.NUM_WORKERS, training=True, distributed=cfg.MODEL.DIST_TRAIN)
            from transformers import AutoProcessor, AutoTokenizer ##chiranjeev - 5 june
            import transformers ##chiranjeev - 5 june
            min_pixels = 56 * 56
            max_pixels = 14 * 14 * 50 * 80
            model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            tokenizer = processor.tokenizer ##chiranjeev - 5 june
            tokenizer = processor.tokenizer ##chiranjeev - 5 june
            new_tokens = ["<second_ans>"]
            existing = tokenizer.special_tokens_map.get("additional_special_tokens", []) ##chiranjeev - 5 june
            updated = list(set(existing + new_tokens))
            tokenizer.add_special_tokens({"additional_special_tokens": updated})
            tokenizer_dataset = tokenizer
        
    # for idx, batch in enumerate(trainloader):
    #     print(f"Batch {idx}:")
    #     for key in batch:
    #         if isinstance(batch[key], torch.Tensor):
    #             print(f"  {key}: shape={batch[key].shape}, dtype={batch[key].dtype}")
    #         else:
    #             print(f"  {key}: type={type(batch[key])}")
    #     break
    # quit()
    
    if not args.qwen_required:
        model = build_model(cfg,dataset.num_train_pids,dataset.num_camera)
    elif args.qwen_required:
        context_model = build_model(cfg,dataset.num_train_pids,dataset.num_camera)
        model = copy.deepcopy(context_model) 
        
    if args.qwen_required:  
        for p in model.parameters():
            p.requires_grad = False
        

    
    if not args.qwen_required:        
        if cfg.TEST.WEIGHT!='':
            print('loading '+cfg.TEST.WEIGHT)
            state_dict = torch.load(cfg.TEST.WEIGHT)
            new_state_dict={}
            for k in state_dict:
                out_k=k.replace('module.','')
                new_state_dict[out_k]=state_dict[k]
            model.load_state_dict(new_state_dict,strict=False)
    
    elif args.qwen_required:
        if cfg.TEST.WEIGHT!='':
            print('loading '+cfg.TEST.WEIGHT)
            state_dict = torch.load(cfg.TEST.WEIGHT)
            new_state_dict={}
            for k in state_dict:
                out_k=k.replace('module.','')
                new_state_dict[out_k]=state_dict[k]
            context_model.load_state_dict(new_state_dict,strict=False)
       
    if not args.qwen_required:
        qwen_model = None ##chiranjeev - if not using qwen model then set to None.
        attention_facial_model = None ##chiranjeev - if not using qwen model then set to None.
        attention_nonbio_model = None ##chiranjeev - 8 july
        idcls_model = None ##chiranjeev - if not using qwen model then set to None.
        context_model = None
        predictor_model = None
      
    elif args.qwen_required:        
        from disent_idiff_attention import CrossAttentionBlockFacial ##chiranjeev
        from disent_jepa_predictor import JEPAPredictor, FusionPredictor ##chiranjeev - 8 july
        embed_dim = 1024
        clip_embed_dim = cfg.MODEL.CLIP_DIM
        qwen_model = build_qwen_model(total_layers=24, rank=32, lora_alpha=8, lora_dropout=0.05) ##chiranjeev - qwen model with lora applied #total_layers=24, rank-512        
        
        attention_facial_model = CrossAttentionBlockFacial(reid_dim=clip_embed_dim, llm_dim=3584, proj_dim=clip_embed_dim, num_heads=4, output_dim=clip_embed_dim) ##chiranjeev - attention facial model
        attention_nonbio_model = CrossAttentionBlockFacial(reid_dim=clip_embed_dim, llm_dim=3584, proj_dim=clip_embed_dim, num_heads=4, output_dim=clip_embed_dim) ##chiranjeev - attention fine model
        
        idcls_model = IDClassifier(embed_dim=embed_dim, num_classes=dataset.num_train_pids) ##chiranjeev - id classifier model
        
        predictor_model = JEPAPredictor(input_dim=clip_embed_dim, hidden_dim=clip_embed_dim//2, output_dim=clip_embed_dim) ##input_dim=1024, hidden_dim=2048, bottleneck_dim=512, output_dim=1024)
        nonbio_predictor_model = JEPAPredictor(input_dim=clip_embed_dim, hidden_dim=clip_embed_dim//2, output_dim=clip_embed_dim) ##input_dim=1024, hidden_dim=2048, bottleneck_dim=512, output_dim=1024)
        
        fusion_predictor_model =  FusionPredictor(embed_dim=clip_embed_dim)
        
    # print("Qwen model trainable parameters:")
    # for name, param in qwen_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # print("Attention Facial model trainable parameters:")
    # for name, param in attention_facial_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # print("ID Classifier model trainable parameters:")
    # for name, param in idcls_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)        
    # print("ReID model trainable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    
    # print("ReID context model trainable parameters:")
    # for name, param in context_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    
    # print("Context predictor model trainable parameters:")
    # for name, param in predictor_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    
    # quit()
    
    from accelerate import Accelerator, DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    if cfg.DATA.DATASET == 'prcc':
        do_train(
            cfg,
            model,
            trainloader,
            accelerator,
            dataset,
            qwen_model, ##chiranjeev - passing the qwen model 
            attention_facial_model, ##chiranjeev - passing the attention facial model
            idcls_model, ##chiranjeev - passing the id_cls model
            context_model, ##chiranjeev - passing the context model
            predictor_model, ##chiranjeev - passing the predictor model
            attention_nonbio_model, ##chiranjeev - 8 july
            nonbio_predictor_model, ##chiranjeev - 8 july
            fusion_predictor_model, ##chiranjeev - 8 july
            args.qwen_required, ##chiranjeev - passing whether qwen model is required or not
            qwen_threshold = cfg.SOLVER.MAX_EPOCHS, #10,
            jepa_threshold = -1, #17,
            val_loader=val_loader,
            val_loader_same=val_loader_same,
            tokenizer_dataset=tokenizer_dataset,
        )
    else:
        do_train(
            cfg,
            model,
            trainloader,
            accelerator,
            dataset,
            qwen_model, ##chiranjeev - passing the qwen model 
            attention_facial_model, ##chiranjeev - passing the attention facial model
            idcls_model, ##chiranjeev - passing the id_cls model
            context_model, ##chiranjeev - passing the context model
            predictor_model, ##chiranjeev - passing the predictor model
            attention_nonbio_model, ##chiranjeev - 8 july
            nonbio_predictor_model, ##chiranjeev - 8 july
            fusion_predictor_model, ##chiranjeev - 8 july
            args.qwen_required, ##chiranjeev - passing whether qwen model is required or not
            qwen_threshold = cfg.SOLVER.MAX_EPOCHS, #10,
            jepa_threshold = -1, #17,
            val_loader=val_loader,
            tokenizer_dataset=tokenizer_dataset,
        )

