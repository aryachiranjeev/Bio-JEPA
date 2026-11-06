import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
# CHANGED: removed `from torch.cuda import amp` and `import torch.distributed as dist`
# CHANGED: added Accelerator import
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import gc
from tqdm import tqdm
from torchvision import transforms


import math
import torch

class CosineMomentumScheduler:
    """
    Schedules the EMA momentum parameter from a base value to a final value
    using a cosine curve over the total number of training steps.
    """
    def __init__(self, base_value=0.996, final_value=1.0, total_steps=100000):
        self.base_value = base_value
        self.final_value = final_value
        self.total_steps = total_steps

    def get_value(self, current_step):
        if current_step >= self.total_steps:
            return self.final_value

        cos_out = math.cos(math.pi * current_step / self.total_steps)
        momentum = self.final_value - 0.5 * (self.final_value - self.base_value) * (1.0 + cos_out)
        return momentum

def cosine_residual_old(F, anchor):
    """
    Remove the component of F along anchor's direction.
    Geometrically correct F - anchor.
    Both [B, dim]. anchor must already have .detach() applied. Here F is projected onto anchor = ||F|| Costheta
    """
    anchor_n   = torch.nn.functional.normalize(anchor, dim=-1)
    proj_len   = (F * anchor_n).sum(dim=-1, keepdim=True)
    F_residual = F - proj_len * anchor_n
    return F_residual

def cosine_residual(F, anchor):
    proj_len = (F * anchor).sum(dim=-1, keepdim=True)
    anchor_sq = (anchor * anchor).sum(dim=-1, keepdim=True) + 1e-8
    F_residual = F - (proj_len / anchor_sq) * anchor
    return F_residual

def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

# CHANGED: removed `local_rank` param, added `accelerator` param
def do_train(cfg,
            model,
            train_loader,
            accelerator,          # CHANGED: replaces local_rank
            dataset,
            qwen_model = None,
            attention_facial_model = None,
            idcls_model = None,
            context_model = None,
            predictor_model = None,
            attention_nonbio_model=None, ##chiranjeev - 8 july
            nonbio_predictor_model=None, ##chiranjeev - 8 july
            fusion_predictor_model=None, ##chiranjeev - 8 july
            # forward_head_context=None,
            # forward_head_target=None,
            qwen_required = False,
            qwen_threshold = 10,
            jepa_threshold = -1, #18,
            val_loader = None,
            val_loader_same = None,
            tokenizer_dataset = None,
            ):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)

    # CHANGED: device comes from accelerator, not hardcoded "cuda"
    device = accelerator.device
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("DIFFER.train")
    logger.info('start training')

    if not qwen_required:
        optimizer, optimizer_qwen, optimizer_center = make_optimizer(cfg, model, center_criterion, qwen_model, attention_facial_model, idcls_model, predictor_model, forward_head_context, forward_head_target)
        scheduler, _ = create_scheduler(cfg, optimizer)
    elif qwen_required:
        optimizer, optimizer_qwen, optimizer_center = make_optimizer(cfg, context_model, center_criterion, qwen_model, attention_facial_model, predictor_model, attention_nonbio_model, nonbio_predictor_model, fusion_predictor_model) #idcls_model
        # optimizer, optimizer_qwen, optimizer_center = make_optimizer(cfg, context_model, center_criterion, qwen_model, attention_facial_model, idcls_model, predictor_model, attention_nonbio_model, nonbio_predictor_model, fusion_predictor_model)
        scheduler, scheduler_qwen = create_scheduler(cfg, optimizer, optimizer_qwen)

    train_loader_batch_size = train_loader.batch_size
    print("train_loader_batch_size:", train_loader_batch_size)

    # CHANGED: replaced manual.to(local_rank) + DistributedDataParallel blocks with accelerator.prepare()
    # Accelerator handles: device placement, DDP wrapping, sampler.set_epoch, LR scaling, mixed precision
    if not qwen_required:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        idcls_model = accelerator.prepare(idcls_model)
    elif qwen_required:
        (model, context_model, qwen_model, attention_facial_model,
         predictor_model, attention_nonbio_model, nonbio_predictor_model, fusion_predictor_model,
         optimizer, optimizer_qwen, scheduler, train_loader) = accelerator.prepare(
            model, context_model, qwen_model, attention_facial_model,
            predictor_model, attention_nonbio_model, nonbio_predictor_model, fusion_predictor_model,
            optimizer, optimizer_qwen, scheduler, train_loader) #idcls_model,
         
        if scheduler_qwen is not None:
            scheduler_qwen = accelerator.prepare(scheduler_qwen)
        for p in model.parameters():
            p.requires_grad = False

    # CHANGED: only main process writes TensorBoard to avoid duplicate logs
    # if accelerator.is_main_process:
    train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATA.DATASET == 'prcc':
        ## with feat_ouput  chiranjeev - 23 july
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        
        ## with x_bio_reid chiranjeev - 23 july
        evaluator_diff_x_bio_reid = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_same_x_bio_reid = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)

    best_rank1 = -np.inf
    best_epoch = 0
    start_train_time = time.time()

    if not qwen_required:
        model.eval()
    elif qwen_required:
        context_model.eval()
        predictor_model.eval()
        nonbio_predictor_model.eval()
        fusion_predictor_model.eval()
        # forward_head_context.eval()

    epoch=0
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()
    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
        
        ## with x_bio_reid chiranjeev - 23 july
        evaluator_diff_x_bio_reid.reset()
        evaluator_same_x_bio_reid.reset()
        
    else:
        evaluator.reset()

    pretrainModelTest=False
    if not qwen_required:
        if pretrainModelTest:
            logger.info("Test the model at the beginning of training")
            if cfg.DATA.DATASET == 'prcc':
                logger.info("Clothes changing setting")
                rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer, qwen_required=qwen_required)
                logger.info("Standard setting")
                test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True, qwen_required=qwen_required)
            elif cfg.DATA.DATASET == 'ltcc':
                logger.info("Clothes changing setting")
                rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True, qwen_required=qwen_required)
                logger.info("Standard setting")
                test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True, qwen_required=qwen_required)
            else:
                rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer, qwen_required=qwen_required)

    elif qwen_required:
        if pretrainModelTest:
            logger.info("Test the model at the beginning of training")
            if cfg.DATA.DATASET == 'prcc':
                ## with feat_ouput  chiranjeev - 23 july
                logger.info("Clothes changing setting  with feat_ouput")
                rank1= test(cfg, context_model, evaluator_diff, val_loader, logger, device,epoch, train_writer, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context
                logger.info("Standard setting  with feat_ouput")
                test(cfg, context_model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context
            
                ## with x_bio_reid  chiranjeev - 23 july
                logger.info("Clothes changing setting  with x_bio_reid")
                rank1= test(cfg, context_model, evaluator_diff_x_bio_reid, val_loader, logger, device,epoch, train_writer, qwen_required=qwen_required, predictor_model=predictor_model, compute_with_x_bio_reid=True) #, forward_head_context=forward_head_context
                logger.info("Standard setting  with x_bio_reid")
                test(cfg, context_model, evaluator_same_x_bio_reid, val_loader_same, logger, device, epoch,  train_writer,test=True, qwen_required=qwen_required, predictor_model=predictor_model, compute_with_x_bio_reid=True) #, forward_head_context=forward_head_context
            
            elif cfg.DATA.DATASET == 'ltcc':
                logger.info("Clothes changing setting")
                rank1 = test(cfg, context_model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True, qwen_required=qwen_required, predictor_model=predictor_model, forward_head_context=forward_head_context)
                logger.info("Standard setting")
                test(cfg, context_model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True, qwen_required=qwen_required, predictor_model=predictor_model, forward_head_context=forward_head_context)
            else:
                rank1= test(cfg, context_model, evaluator, val_loader, logger, device,epoch,train_writer, qwen_required=qwen_required, predictor_model=predictor_model, forward_head_context=forward_head_context)


    if qwen_required:
        @torch.no_grad()
        def update_target_ema(context_model, target_model, momentum):
            """
            In-place update of target model weights using Exponential Moving Average.
            θ_target = m * θ_target + (1 - m) * θ_context
            """
            # CHANGED: use accelerator.unwrap_model instead of checking for .module attribute
            ctx_modules = accelerator.unwrap_model(context_model)
            tgt_modules = accelerator.unwrap_model(target_model)

            for param_ctx, param_tgt in zip(ctx_modules.parameters(), tgt_modules.parameters()):
                if param_ctx.requires_grad:
                    param_tgt.data.mul_(momentum).add_(param_ctx.data, alpha=1.0 - momentum)

        total_training_steps = epochs * len(train_loader)
        momentum_scheduler = CosineMomentumScheduler(
            base_value=0.996,
            final_value=1.0,
            total_steps=total_training_steps
        )
        global_step = 0
        
    ## chiranjeev - starting here 10 july
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        if cfg.DATA.DATASET == 'ltcc':
            evaluator_diff.reset()
            evaluator_general.reset()
        elif cfg.DATA.DATASET == 'prcc':
            evaluator_diff.reset()
            evaluator_same.reset()
            ## with x_bio_reid chiranjeev - 23 july
            evaluator_diff_x_bio_reid.reset()
            evaluator_same_x_bio_reid.reset()
        else:
            evaluator.reset()

        # CHANGED: removed sampler.set_epoch — accelerator.prepare(train_loader) handles this automatically
        scheduler.step(epoch) ##chiranjeev - 23 july
        if not qwen_required:
            model.train()

        elif qwen_required:
            context_model.train()
            predictor_model.train()
            # forward_head_context.train()
            attention_nonbio_model.train()
            nonbio_predictor_model.train()
            fusion_predictor_model.train()
            # idcls_model.train() ##chiranjeev - 8 july

            qwen_model.train()
            attention_facial_model.train()
            # forward_head_target.train()
            model.eval()

            # if scheduler_qwen is not None and epoch <= qwen_threshold  and idx % 4==0:
            #     scheduler_qwen.step(epoch)

        for idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            with accelerator.accumulate(context_model, qwen_model, attention_facial_model, predictor_model, attention_nonbio_model, nonbio_predictor_model, fusion_predictor_model): #idcls_model, forward_head_context, forward_head_target):            
            
                step=idx+len(train_loader)*epoch

                if not qwen_required:
                    samples=data['image']
                    targets=data['pid']
                    camids=data['camid']
                    clothes=data['clothes_id'] if 'clothes_id' in data else None
                    caption_feature=data['caption_feature'] if 'caption_feature' in data else None

                elif qwen_required:
                    import sys
                    sys.path.append('./data')
                    import data.img_transforms as T
                    # reid_aug = T.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0)), T.RandomErasing(probability=0.65)])
                    samples_context = data['image'] #reid_aug(data['image'])
                    camids_context = data['camid']
                    samples = data['image']
                    targets=data['pid']
                    camids=data['camid']
                    clothes=data['clothes_id'] if 'clothes_id' in data else None
                    caption_feature=data['caption_feature'] if 'caption_feature' in data else None

                optimizer.zero_grad()
                if optimizer_center is not None:
                    optimizer_center.zero_grad()

                if optimizer_qwen is not None:
                    if (epoch <= qwen_threshold) and (idx % 4==0):
                        optimizer_qwen.zero_grad()

                if not qwen_required:
                    score, embed_probe, _ = model(samples, cam_label=camids, qwen_required=qwen_required)
                    # cls_score = idcls_model(embed_probe)

                elif qwen_required:
                    ## embed_probe_context_trip-(B,1024), embed_probe_target_trip - (B, 1024), embed_probe_context_tokens - (B, N, 512), embed_probe_target_tokens - (B, N, 512) ##chiranjeev - 8 july
                    score, embed_probe_context_trip, embed_probe_context_tokens, x_bio_reid = context_model(samples_context, cam_label=camids_context, qwen_required=qwen_required)
                    with torch.no_grad():
                        _, embed_probe_target_trip, embed_probe_target_tokens, _ = model(samples, cam_label=camids, qwen_required=qwen_required)

                if qwen_required:
                    # CHANGED: removed `with torch.amp.autocast(...)` — accelerator handles mixed precision
                    # Configure it at Accelerator init: Accelerator(mixed_precision="bf16")
                    outputs = qwen_model(
                        input_ids=data['input_ids'],
                        attention_mask=data['attention_mask'],
                        pixel_values=data['pixel_values'],
                        image_grid_thw=data['image_grid_thw'],
                        labels=data['labels'],
                        return_dict=True,
                        output_hidden_states=True,
                        labels_facial=data['labels_facial'],
                        test = False,
                        use_cache=False,
                    )

                    if idx == 10 or idx == 50 or idx == 100 or idx == 200:
                        tokenizer = tokenizer_dataset
                        pred_ids_fine = torch.argmax(outputs.logits, dim=-1)
                        pred_ids_facial = torch.argmax(outputs.logits_facial, dim=-1)

                        fine_texts, facial_texts = [], []
                        for b in range(pred_ids_fine.shape[0]):
                            fine_mask = data['labels'][b] != -100
                            logit_mask = torch.zeros_like(fine_mask)
                            logit_mask[:-1] = fine_mask[1:]
                            fine_texts.append(tokenizer.decode(pred_ids_fine[b][logit_mask], skip_special_tokens=True))

                            facial_mask = data['labels_facial'][b] != -100
                            logit_mask_f = torch.zeros_like(facial_mask)
                            logit_mask_f[:-1] = facial_mask[1:]
                            facial_texts.append(tokenizer.decode(pred_ids_facial[b][logit_mask_f], skip_special_tokens=True))

                        print("fine_texts:", fine_texts)
                        print("facial_texts:", facial_texts)


                    target_selective_mask = [tar != -100 for tar in data['labels']]
                    target_facial_selective_mask = [tarf != -100 for tarf in data['labels_facial']]
                    hidden_facial_states_attn_unpad_output = []
                    text_token_lens = []
                    for i in range(outputs.hidden_facial_states.shape[0]):
                        ans = (outputs.hidden_facial_states[i].unsqueeze(0)[:,target_facial_selective_mask[i],:])[:,:-1,:]
                        hidden_facial_states_attn_unpad_output.append(ans.squeeze(0))
                        text_token_lens.append(ans.shape[1])
                    max_len_facial = max(text_token_lens)
                    hidden_facial_states_attn = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_len_facial - t.size(0))) for t in hidden_facial_states_attn_unpad_output])
                    text_mask_facial = torch.zeros(hidden_facial_states_attn.shape[0], max_len_facial, dtype=torch.bool).to(outputs.hidden_facial_states.device)
                    for i, l in enumerate(text_token_lens):
                        text_mask_facial[i, :l] = 1

                    hidden_fine_states_attn_unpad_output = []
                    text_token_lens = []
                    for i in range(outputs.hidden_fine_states.shape[0]):
                        ans = (outputs.hidden_fine_states[i].unsqueeze(0)[:,target_selective_mask[i],:])[:,2:-1,:]
                        hidden_fine_states_attn_unpad_output.append(ans.squeeze(0))
                        text_token_lens.append(ans.shape[1])
                    max_len_fine = max(text_token_lens)
                    hidden_fine_states_attn = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_len_fine - t.size(0))) for t in hidden_fine_states_attn_unpad_output])
                    text_mask_fine = torch.zeros(hidden_fine_states_attn.shape[0], max_len_fine, dtype=torch.bool).to(outputs.hidden_fine_states.device)
                    for i, l in enumerate(text_token_lens):
                        text_mask_fine[i, :l] = 1

                    # embed_probe_context = embed_probe_context.detach()
                    fine_probe, nonbio_mask_probe_embed_tokens = attention_nonbio_model(embed_probe_context_tokens, hidden_fine_states_attn.float().detach(), text_mask_fine, bio = False)
                    facial_probe, facial_mask_probe_embed_tokens = attention_facial_model(embed_probe_context_tokens, hidden_facial_states_attn.float().detach(), text_mask_facial)
                    # fine_probe = forward_head_target(fine_probe)
                    # facial_probe = forward_head_target(facial_probe)

                    # embed_probe_context_pred = predictor_model(embed_probe_context)
                    # embed_probe_context_pred = forward_head_context(embed_probe_context_pred)
                    # cls_score = idcls_model(embed_probe_context_pred)
                    
                    bio_pred_probe_embed_tokens = predictor_model(facial_mask_probe_embed_tokens) ##chiranjeev - 8 july
                    nonbio_pred_probe_embed_tokens = nonbio_predictor_model(nonbio_mask_probe_embed_tokens) ##chiranjeev - 8 july
                    holistic_embed_probe_context_pred_tokens = fusion_predictor_model(bio_pred_probe_embed_tokens, nonbio_pred_probe_embed_tokens)  ##chiranjeev - 8 july
                                        
                    # cls_score = idcls_model(embed_probe_context) ##chiranjeev - 8 july
                    # score = {'cls_score':cls_score.to(facial_probe.device)}  ##chiranjeev - 8 july
                    # cls_score = cls_score.to(facial_probe.device) ##chiranjeev - 8 july
                    targets = targets.to(accelerator.device)
                    camids = camids.to(accelerator.device)

                    ##chiranjeev done till here 10 july

                    llm_loss = outputs.loss.float()
                    llm_loss_facial = outputs.loss_facial.float()
                    llm_loss_fine = outputs.loss_fine.float()

                    is_qwen_threshold = epoch <= qwen_threshold
                    is_jepa_threshold = epoch > jepa_threshold

                    loss = loss_fn(score, embed_probe_context_trip, targets, camids,caption_feature,clothes_ids=clothes,train_writer=train_writer,step=step, qwen_required=qwen_required, facial_probe=facial_probe, fine_probe=fine_probe, embed_probe_context_pred=holistic_embed_probe_context_pred_tokens, embed_probe_target=embed_probe_target_tokens , llm_loss=llm_loss, llm_loss_facial=llm_loss_facial, llm_loss_fine=llm_loss_fine, is_qwen_threshold=is_qwen_threshold, is_jepa_threshold=is_jepa_threshold) ##chiranjeev - 8 july
                    # loss = loss_fn(score, embed_probe_context_trip, targets, camids,caption_feature,clothes_ids=clothes,train_writer=train_writer,step=step, qwen_required=qwen_required, facial_probe=facial_probe, fine_probe=fine_probe, embed_probe_context_pred=embed_probe_context_pred, llm_loss=llm_loss, llm_loss_facial=llm_loss_facial, llm_loss_fine=llm_loss_fine, is_qwen_threshold=is_qwen_threshold, is_jepa_threshold=is_jepa_threshold)

                elif not qwen_required:
                    loss = loss_fn(score, embed_probe, targets, camids,caption_feature,clothes_ids=clothes,train_writer=train_writer,step=step)
                    
                current_lr=scheduler._get_lr(epoch)[0] ##chiranjeev - 23 july
                if accelerator.is_main_process:
                    train_writer.add_scalar('loss/total', loss.item(), step)
                    train_writer.add_scalar('lr', current_lr, step)
                
                
                # CHANGED: replaced loss.backward() with accelerator.backward(loss)
                # This handles gradient scaling for mixed precision automatically
                accelerator.backward(loss)

                optimizer.step()
                
                if optimizer_qwen is not None:
                    if (epoch <= qwen_threshold) and (idx%4 == 0):
                        torch.nn.utils.clip_grad_norm_(qwen_model.parameters(), max_norm=1.0)
                        optimizer_qwen.step()

                if 'center' in cfg.MODEL.LOSS_TYPE:
                    print("center")
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    optimizer_center.step()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == targets).float().mean()
                elif isinstance(score, dict):
                    if isinstance(score['cls_score'], list):
                        acc = (score['cls_score'][0].max(1)[1] == targets).float().mean()
                    else:
                        acc = (score['cls_score'].max(1)[1] == targets).float().mean()
                else:
                    acc = (score.max(1)[1] == targets).float().mean()

                loss_meter.update(loss.item(), samples.shape[0])
                acc_meter.update(acc, 1)

                if qwen_required:
                    current_momentum = momentum_scheduler.get_value(global_step)
                    update_target_ema(context_model, model, current_momentum) ##chiranjeev - 8 july
                    global_step += 1

                # CHANGED: removed torch.cuda.synchronize() — not needed with accelerator
                if (idx + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (idx + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, current_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, train_loader_batch_size / time_per_batch))

        # CHANGED: wrap saves in accelerator.is_main_process + accelerator.unwrap_model() for correct state_dict keys
        # if accelerator.is_main_process:
        # '''
        if (not qwen_required) and (epoch % 2 == 0):
            # save_dir = f'/home/aac/chiranjeev/PRCC_Differ/output/ltcc/{epoch}/' ##chiranjeev - 23 july
            save_dir = f'/home/aac/chiranjeev/PRCC_Differ/output/prcc/{epoch}/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_dir, cfg.MODEL.NAME + f'.pth'))
            print("SAVED models at:", save_dir)

        elif (qwen_required) and (epoch % 2 == 0):
            # save_dir = f'/home/aac/chiranjeev/PRCC_Differ/output/ltcc/{epoch}/' ##chiranjeev - 23 july
            save_dir = f'/home/aac/chiranjeev/PRCC_Differ/output/prcc/{epoch}/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(attention_facial_model).state_dict(), os.path.join(save_dir, 'attention_facial_model.pth'))
            # torch.save(accelerator.unwrap_model(idcls_model).state_dict(), os.path.join(save_dir, 'idcls_model.pth'))
            torch.save(accelerator.unwrap_model(qwen_model).state_dict(), os.path.join(save_dir, 'qwen_model.pth'))
            torch.save(accelerator.unwrap_model(context_model).state_dict(), os.path.join(save_dir, cfg.MODEL.NAME + '_context_model.pth'))
            torch.save(accelerator.unwrap_model(predictor_model).state_dict(), os.path.join(save_dir, 'predictor_model.pth'))
            # torch.save(accelerator.unwrap_model(forward_head_context).state_dict(), os.path.join(save_dir, 'forward_head_context.pth'))
            # torch.save(accelerator.unwrap_model(forward_head_target).state_dict(), os.path.join(save_dir, 'forward_head_target.pth'))
            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_dir, 'target_model.pth'))
            torch.save(accelerator.unwrap_model(attention_nonbio_model).state_dict(), os.path.join(save_dir, 'attention_nonbio_model.pth'))
            torch.save(accelerator.unwrap_model(nonbio_predictor_model).state_dict(), os.path.join(save_dir, 'nonbio_predictor_model.pth'))
            torch.save(accelerator.unwrap_model(fusion_predictor_model).state_dict(), os.path.join(save_dir, 'fusion_predictor_model.pth'))
            print("SAVED models at:", save_dir)
        # '''

        if epoch % eval_period == 0:
            if not qwen_required:
                model.eval()
                if cfg.DATA.DATASET == 'prcc':
                    logger.info("Clothes changing setting")
                    rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer, qwen_required=qwen_required)
                    logger.info("Standard setting")
                    test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True, qwen_required=qwen_required)
                elif cfg.DATA.DATASET == 'ltcc':
                    logger.info("Clothes changing setting")
                    rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True, qwen_required=qwen_required)
                    logger.info("Standard setting")
                    test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True, qwen_required=qwen_required)
                else:
                    rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer, qwen_required=qwen_required)

            elif qwen_required:
                context_model.eval()
                predictor_model.eval()
                # forward_head_context.eval()
                if cfg.DATA.DATASET == 'prcc':
                    ## with feat_ouput  chiranjeev - 23 july
                    logger.info("Clothes changing setting with feat_ouput")
                    rank1= test(cfg, context_model, evaluator_diff, val_loader, logger, device,epoch, train_writer, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context)
                    logger.info("Standard setting with feat_ouput")
                    test(cfg, context_model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context)
                
                    ## with x_bio_reid  chiranjeev - 23 july
                    logger.info("Clothes changing setting  with x_bio_reid")
                    rank1= test(cfg, context_model, evaluator_diff_x_bio_reid, val_loader, logger, device,epoch, train_writer, qwen_required=qwen_required, predictor_model=predictor_model, compute_with_x_bio_reid=True) #, forward_head_context=forward_head_context)
                    logger.info("Standard setting  with x_bio_reid")
                    test(cfg, context_model, evaluator_same_x_bio_reid, val_loader_same, logger, device, epoch,  train_writer,test=True, qwen_required=qwen_required, predictor_model=predictor_model, compute_with_x_bio_reid=True) #, forward_head_context=forward_head_context)
            
                elif cfg.DATA.DATASET == 'ltcc':
                    logger.info("Clothes changing setting")
                    rank1 = test(cfg, context_model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context)
                    logger.info("Standard setting")
                    test(cfg, context_model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context)
                else:
                    rank1= test(cfg, context_model, evaluator, val_loader, logger, device,epoch,train_writer, qwen_required=qwen_required, predictor_model=predictor_model) #, forward_head_context=forward_head_context)

            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch
                logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
                # CHANGED: replaced dist.get_rank() == 0 with accelerator.is_main_process
                # CHANGED: unwrap_model() to get clean state_dict without DDP module. prefix
                if accelerator.is_main_process:
                    if not qwen_required:
                        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    elif qwen_required:
                        torch.save(accelerator.unwrap_model(context_model).state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("Save the best model")

    # CHANGED: replaced dist.get_rank() == 0 with accelerator.is_main_process
    if accelerator.is_main_process:
        logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same = None,
                 qwen_required = False,
                 ):
    logger = logging.getLogger("DIFFER.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()
    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if cfg.DATA.DATASET == 'prcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,clothChanging=True, qwen_required=qwen_required)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True, qwen_required=qwen_required)
    elif cfg.DATA.DATASET == 'ltcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,ltcc=True,clothChanging=True, qwen_required=qwen_required)
        logger.info("Standard setting")
        test(cfg, model, evaluator_general, val_loader, logger, device, test=True, qwen_required=qwen_required)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True,clothChanging=True, qwen_required=qwen_required)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, test_writer=None,test=False,ltcc=False,clothChanging=False, qwen_required=True, predictor_model=None, compute_with_x_bio_reid=False): #, forward_head_context=None):
    for n_iter, data in enumerate(val_loader):
        if n_iter%100==0:
            logger.info(f"processing batch {n_iter}/{len(val_loader)}")

        imgs=data['image']
        pids=data['pid']
        camids=data['camid']
        clothes_id=data['clothes_id']
        img_paths=data['image_path']

        with torch.no_grad():

            imgs = imgs.to(device)
            camids0=camids.to(device)

            if not qwen_required:
                _, _, embed_probe, _ = model(imgs, cam_label=camids0, qwen_required=qwen_required)
                feat = embed_probe
            elif qwen_required: # embed_probe_context
                _, embed_probe_context_trip, embed_probe_context_tokens, x_bio_reid = model(imgs, cam_label=camids0, qwen_required=qwen_required)
                # print("embed_probe_context_trip.shape:", embed_probe_context_trip[0].shape)
                # feat = predictor_model(embed_probe_context)
                # feat = forward_head_context(feat)
                if compute_with_x_bio_reid:
                    feat = x_bio_reid.mean(dim=1)
                else: 
                    feat = embed_probe_context_trip[0]

            if ltcc:
                evaluator.update((feat, pids, camids,clothes_id,img_paths))
            else:
                evaluator.update((feat, pids, camids,clothes_id,img_paths))


    cmc, mAP, _, _, _, queryFeature, galleryFeature,imgpath = evaluator.compute()
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("mAP-:{:.1%}".format(mAP))


    rank1 = cmc[0]
    if test :
        torch.cuda.empty_cache()
        return rank1


    logger.info("Validation Results - Epoch: {}".format(epoch))

    test_writer.add_scalar('rank1', rank1, epoch)
    test_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1
