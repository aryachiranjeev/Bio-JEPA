import torch

'''
def _collect_params(cfg, model, lr_new):
    params = []
    large_lr_layers = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
        
    return params, large_lr_layers


def make_optimizer(cfg, model, center_criterion, qwen_model=None, attention_facial_model=None, predictor_model=None, attention_nonbio_model=None, nonbio_predictor_model=None, fusion_predictor_model=None, lr_new=0):

    # --- ReID model + attention + idcls → existing optimizer (SGD or AdamW per config) ---
    reid_params, large_lr_layers = _collect_params(cfg, model, lr_new)

    if attention_facial_model is not None:
        p, ll = _collect_params(cfg, attention_facial_model, lr_new)
        reid_params += p
        large_lr_layers += ll
    
    if predictor_model is not None:
        p, ll = _collect_params(cfg, predictor_model, lr_new)
        reid_params += p
        large_lr_layers += ll
        
    if attention_nonbio_model is not None:
        p, ll = _collect_params(cfg, attention_nonbio_model, lr_new)
        reid_params += p
        large_lr_layers += ll
      
    if nonbio_predictor_model is not None:
        p, ll = _collect_params(cfg, nonbio_predictor_model, lr_new)
        reid_params += p
        large_lr_layers += ll
        
    if fusion_predictor_model is not None:
        p, ll = _collect_params(cfg, fusion_predictor_model, lr_new)
        reid_params += p
        large_lr_layers += ll
    
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(reid_params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(reid_params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, eps=1e-6, betas=(0.9, 0.999))

    # --- Qwen (LLM + LoRA) → AdamW with separate LR ---
    optimizer_qwen = None
    if qwen_model is not None:
        qwen_params = [
            {"params": value, "lr": cfg.SOLVER.QWEN_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
            for _, value in qwen_model.named_parameters()
            if value.requires_grad
        ]
        # eps=1e-6 avoids NaN in Adam update when bf16 gradients have limited precision
        optimizer_qwen = torch.optim.AdamW(qwen_params, lr=cfg.SOLVER.QWEN_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, eps=1e-6, betas=(0.9, 0.999))

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    # optimizer_qwen = None ##5july
    return optimizer, optimizer_qwen, optimizer_center

'''

def _collect_params(cfg, model, lr_new):
    params = []
    large_lr_layers = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if lr_new==0:
            lr = cfg.SOLVER.BASE_LR
        else:
            lr=lr_new
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "head_cls" in key:
                print("I am in [Make_Optimizer], cfg.SOLVER.LARGE_FC_LR, head_cls in key")
                large_lr_layers.append(key)
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LARGE_FC_FACTOR
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
    return params, large_lr_layers


def make_optimizer(cfg, model, center_criterion, qwen_model=None, attention_facial_model=None, predictor_model=None, attention_nonbio_model=None, nonbio_predictor_model=None, fusion_predictor_model=None, lr_new=0):
    
    # --- ReID model + attention + idcls → existing optimizer (SGD or AdamW per config) ---
    reid_params, large_lr_layers = _collect_params(cfg, model, lr_new)

    if attention_facial_model is not None:
        p, ll = _collect_params(cfg, attention_facial_model, lr_new)
        reid_params += p
        large_lr_layers += ll
    
    if predictor_model is not None:
        p, ll = _collect_params(cfg, predictor_model, lr_new)
        reid_params += p
        large_lr_layers += ll
        
    if attention_nonbio_model is not None:
        p, ll = _collect_params(cfg, attention_nonbio_model, lr_new)
        reid_params += p
        large_lr_layers += ll
      
    if nonbio_predictor_model is not None:
        p, ll = _collect_params(cfg, nonbio_predictor_model, lr_new)
        reid_params += p
        large_lr_layers += ll
        
    if fusion_predictor_model is not None:
        p, ll = _collect_params(cfg, fusion_predictor_model, lr_new)
        reid_params += p
        large_lr_layers += ll
    
    if len(large_lr_layers) > 0:
        print(f'Using {cfg.SOLVER.LARGE_FC_FACTOR} times learning rate for ', large_lr_layers)
    
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(reid_params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(reid_params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(reid_params)


    # --- Qwen (LLM + LoRA) → AdamW with separate LR ---
    optimizer_qwen = None
    if qwen_model is not None:
        qwen_params = [
            {"params": value, "lr": cfg.SOLVER.QWEN_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
            for _, value in qwen_model.named_parameters()
            if value.requires_grad
        ]
        # eps=1e-6 avoids NaN in Adam update when bf16 gradients have limited precision
        optimizer_qwen = torch.optim.AdamW(qwen_params, lr=cfg.SOLVER.QWEN_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, eps=1e-6)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    # optimizer_qwen = None ##5july
    return optimizer, optimizer_qwen, optimizer_center

# '''