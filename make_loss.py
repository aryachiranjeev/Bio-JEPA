# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss,euclidean_dist
from .center_loss import CenterLoss
from .arcface import CrossEntropy

import torch.distributed as dist

# def reduce_loss(loss):
#     """Average loss across all GPUs for logging."""
#     reduced = loss.detach().clone()
#     dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
#     reduced /= dist.get_world_size()  # 5 GPUs
#     return reduced.item()


mse_loss = torch.nn.MSELoss()

def jepa_mse_loss(embed_probe_context_pred, facial_probe_stop_grad):
    """JEPA loss to align facial probe with context prediction."""
    jepa_loss = F.smooth_l1_loss(embed_probe_context_pred, facial_probe_stop_grad)
    # jepa_loss = AllReduce.apply(jepa_loss)
    # jepa_loss = mse_loss(embed_probe_context_pred, facial_probe_stop_grad)
    return jepa_loss

def orthogonality_loss(f_b, f_non_b):
    """Force biometric and non-biometric subspaces to be orthogonal."""
    B, N, D = f_b.shape
    # (B*N, D)
    f_b = f_b.reshape(B * N, D)
    f_non_b = f_non_b.reshape(B * N, D)
    f_bio_n  = torch.nn.functional.normalize(f_b,     dim=-1)   # [B, N, 512]
    f_non_n  = torch.nn.functional.normalize(f_non_b, dim=-1)   # [B, N, 512]
    cross    = f_bio_n.T @ f_non_n / (B*N)          # [512, 512]
    return torch.linalg.matrix_norm(cross, ord='fro')

def contrastive_disentangle_loss(f, f_bio, f_non_bio, margin=0.5):
    """
    Pull  : f_gal and f_bio should be similar (F contains biometric signal)
    Push  : f_bio and f_non_bio should be dissimilar (orthogonal subspaces)
    """
    temperature = 4.0  # soften distributions to reveal dark knowledge
    P_f = torch.nn.functional.log_softmax(f / temperature, dim=-1)
    P_bio = torch.nn.functional.softmax(f_bio / temperature, dim=-1)
    L_pull = torch.nn.functional.kl_div(P_f, P_bio.detach(), reduction='batchmean') * (temperature ** 2)
    
    # L_pull = mse_loss(f, f_bio.detach())
    L_push = torch.tensor(0.0).to(f.device)
    
    '''
    L_pull = mse_loss(f, f_bio.detach())
    L_push = torch.tensor(0.0).to(f.device)
    '''
    # L_pull = (1 - torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(f, dim=-1), torch.nn.functional.normalize(f_bio, dim=-1))).mean()
    # L_push = torch.linalg.matrix_norm((torch.nn.functional.normalize(f, dim=-1).T @ torch.nn.functional.normalize(f_non_bio, dim=-1))/f.size(0), ord='fro')
        
    
    # euclidean_distance_min = torch.nn.functional.pairwise_distance(f_gal, f_bio.detach())
    # L_pull = torch.mean(torch.pow(euclidean_distance_min, 2))
    # euclidean_distance_max = torch.nn.functional.pairwise_distance(f_gal, f_non_bio.detach())
    # L_push = torch.mean(torch.pow(torch.clamp(1.0 - euclidean_distance_max, min=0.0), 2))
    
    # Pull — minimize distance between vision features and biometric residual
    # dist_pos = 1 - torch.nn.functional.cosine_similarity(f_gal, f_bio)
    # L_pull   = dist_pos.pow(2).mean()

    # Push — maximize distance between bio and non-bio residuals
    # dist_neg = 1 - torch.nn.functional.cosine_similarity(f_bio, f_non_bio)
    # L_push   = torch.clamp(margin - dist_neg, min=0.0).pow(2).mean()

    return L_pull, L_push


def uniformity_loss(F_residual):
    """Non-bio residual should scatter uniformly — penalise clustering."""
    z      = torch.nn.functional.normalize(F_residual, dim=-1)
    sq_dist = torch.cdist(z, z, p=2).pow(2)
    return torch.log(torch.exp(-2 * sq_dist).mean())


def jepa_infonce_loss(embed_probe_context_pred, facial_probe_stop_grad, temperature = 0.07):
    """
    Drop-in replacement for your jepa_mse_loss function.
 
    Usage in make_loss.py:
        # Old:
        # jepa_loss = jepa_mse_loss(embed_probe_context_pred, facial_probe.detach())
        # New:
        jepa_loss, jepa_acc = jepa_infonce_loss(embed_probe_context_pred, facial_probe)
 
    Args:
        f_pred:      (B, D) predictor output — gradients flow through this.
        f_bio:       (B, D) bio features — gets detached inside.
        temperature: Scaling temperature.
 
    Returns:
        loss:  Scalar loss.
        acc:   Scalar accuracy for logging.
    """
    # facial_probe_stop_grad = facial_probe_stop_grad.detach()
 
    f_pred = F.normalize(embed_probe_context_pred, dim=-1)
    f_bio = F.normalize(facial_probe_stop_grad, dim=-1)
 
    logits = f_pred @ f_bio.T / temperature  # (B, B)
    labels = torch.arange(f_pred.shape[0], device=f_pred.device)
    
    loss = F.cross_entropy(logits, labels)
    # loss = (
    #     F.cross_entropy(logits, labels)
    #     + F.cross_entropy(logits.T, labels)
    # ) * 0.5
    
    return loss


def clip_contrastive_loss(image_features, text_features, logit_scale=1.0):
    logits_per_image = logit_scale*image_features  @ text_features.T
    labels = torch.arange(image_features.shape[0],device=image_features.device)
    loss= F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc

def clip_contrastive_score_loss(score, logit_scale=torch.zeros(1)):
    logit_scale=logit_scale.exp().to(score.device)
    logits_per_image = logit_scale*score
    labels = torch.arange(score.shape[0],device=score.device)
    loss= F.cross_entropy(logits_per_image, labels)
    
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc

def clip_sigmoid_loss(image_features, text_features, logit_scale=1.0,logit_bias=0.0):
    logit_scale=logit_scale.type(image_features.type())
    logit_bias=logit_bias.type(image_features.type())
    logits_per_image =logit_scale*image_features  @ text_features.T+logit_bias
    labels = 2*torch.eye(image_features.shape[0],device=image_features.device)-1
    #binary cross entropy
    loss = F.binary_cross_entropy_with_logits(logits_per_image,labels)
    acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
    return loss,acc
    
def clip_l2_loss(score, logit_scale=torch.zeros(1)):
    #diagnal scores
    positive_score=score.diag()
    loss=(1-positive_score).sum()/score.shape[0]
   
    labels = torch.arange(score.shape[0],device=score.device)
    acc = (score.argmax(-1) == labels).sum() / score.shape[0]
    return loss,acc


def make_loss(cfg, num_classes,lossTypes=None):    # modified by gu
    if lossTypes is None:
        lossTypes=cfg.MODEL.LOSS_TYPE

    sampler = cfg.DATA.SAMPLER
   
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.DATA.SAMPLER:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(lossTypes))
   
    
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent=torch.nn.CrossEntropyLoss()
        
        
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, caption_feature,clothes_ids,train_writer,step, qwen_required=False, facial_probe=None, fine_probe=None, embed_probe_context_pred=None, embed_probe_target=None, llm_loss=None, llm_loss_facial=None, llm_loss_fine=None, is_qwen_threshold=False, is_jepa_threshold=False):
            jepa_loss_weight = 10.0 #cfg.MODEL.ID_LOSS_WEIGHT * 2.0 #10.0 #20.0
            ortho_loss_weight = cfg.MODEL.ID_LOSS_WEIGHT * 5.0 #10.0
            loss=torch.tensor(0.0,device=feat[0].device)
            if 'ce' in lossTypes:
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                    id_acc = (score[0].argmax(-1) == target).sum() / len(target)
                elif isinstance(score, dict):
                    if isinstance(score['cls_score'], list):
                        ID_LOSS = [xent(s, target) for s in score['cls_score']]
                       
                        for i in range(len(score['cls_score'])):
                            train_writer.add_scalar('loss/id_'+str(i), ID_LOSS[i].item(), step)
                            id_acc = (score['cls_score'][i].argmax(-1) == target).sum() / len(target)
                            train_writer.add_scalar('acc/id_'+str(i), id_acc.item(), step)
                        ID_LOSS = sum(ID_LOSS)
                        id_acc = (score['cls_score'][1].argmax(-1) == target).sum() / len(target)
                    else:
                        ID_LOSS = xent(score['cls_score'], target)
                        id_acc = (score['cls_score'].argmax(-1) == target).sum() / len(target)
                else:
                    ID_LOSS = xent(score, target)
                    id_acc = (score.argmax(-1) == target).sum() / len(target)
                train_writer.add_scalar('loss/id', ID_LOSS.item(), step)
                train_writer.add_scalar('acc/id', id_acc.item(), step)
                loss+=cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS 
                
            if 'triplet' in lossTypes:
                if isinstance(feat, list):
                    #TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                    TRI_LOSS = triplet(feat[0], target)[0]
                    #TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                train_writer.add_scalar('loss/triplet', TRI_LOSS.item(), step)
                loss+=cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                
            # if 'clipBio' in lossTypes  or 'clipBioReverse' in lossTypes: ##chiranjeev - 8 july
            if not('clipBio' in lossTypes  or 'clipBioReverse' in lossTypes): ##chiranjeev - 8 july
                batch_size=feat[0].shape[0]
                if cfg.MODEL.LAST_LAYER in ['transformer']:
                    image_features_bio,image_features_nonbio=feat
                elif cfg.MODEL.LAST_LAYER in ['fc']:
                    image_features_bio,image_features_nonbio,weight_bio,weight_nonbio=feat
                elif cfg.MODEL.LAST_LAYER in ['clipFc','clipMLP']:
                    image_features_bio=score['clip_bio_score']    
                    if 'clip_nonbio_score' in score:
                        image_features_nonbio=score['clip_nonbio_score']
                    else:
                        image_features_nonbio=None
                # if 'cip_bio_reverse_score' in score:
                #     image_feature_reverse=score['clip_bio_reverse_score']
                #     #

                image_features_bio=torch.nn.functional.normalize(image_features_bio,dim=-1)
                if image_features_nonbio is not None:
                    image_features_nonbio=torch.nn.functional.normalize(image_features_nonbio,dim=-1)
                if image_features_bio.shape[1]>200:
                    image_features_bio=image_features_bio.unsqueeze(1)
                # if caption_feature.shape[1]>200:
                #     caption_feature=caption_feature.unsqueeze(1)
                        
                caption_feature=caption_feature.type(image_features_bio.type())
                # print("caption_feature:",caption_feature.shape)
                text_features_bio=caption_feature[:,:image_features_bio.shape[1]]
                text_features_bio=torch.nn.functional.normalize(text_features_bio,dim=-1)
                text_features_nonbio=caption_feature[:,image_features_bio.shape[1]:]
                text_features_nonbio=torch.nn.functional.normalize(text_features_nonbio,dim=-1)
                #text_features_bio.norm(dim=-1, keepdim=True)
            
            
            # if 'clipBio' in lossTypes: ##chiranjeev - 8 july
            if not ('clipBio' in lossTypes):  ##chiranjeev - 8 july
                loss_clip_all=0
                for i in range(image_features_bio.shape[1]):
                    logits_per_bio = image_features_bio[:,i]  @ text_features_bio[:,i].T
                    labels = torch.arange(image_features_bio[:,i].shape[0],device=image_features_bio.device)
                    loss_clip = F.cross_entropy(logits_per_bio, labels)
                   
                    i2t_acc = (logits_per_bio.argmax(-1) == labels).sum() / len(logits_per_bio)
                   
                        
                    loss_clip_all+=loss_clip
                    train_writer.add_scalar('acc/clip_bio_'+str(i), i2t_acc.item(), step)
                        
                    #loss_clip = loss_image 
                    train_writer.add_scalar('loss/clip_bio_'+str(i), loss_clip.item(), step)
                    #print(loss_clip)
               
                loss+=loss_clip
                
            # if 'clipBioReverse' in lossTypes : ##chiranjeev - 8 july
            if not ('clipBioReverse' in lossTypes): ##chiranjeev - 8 july
                image_feature_reverse=score['clip_bio_reverse_score']
                loss_clip_reverse=0
              
                for i in range(image_feature_reverse.shape[1]):
                    image_feature_reverse_i=torch.nn.functional.normalize(image_feature_reverse[i])
                    image_feature_reverse_i=image_feature_reverse_i.type(text_features_nonbio.type())
                    logits_per_bio_reverse = image_feature_reverse_i @ text_features_nonbio[:,i].T
                    # symmetric loss function
                    labels = torch.arange(image_feature_reverse[i].shape[0],device=image_feature_reverse[i].device)
                    loss_image = F.cross_entropy(logits_per_bio_reverse, labels)
                    
                    i2t_acc = (logits_per_bio_reverse.argmax(-1) == labels).sum() / len(logits_per_bio_reverse)
                    train_writer.add_scalar('acc/clip_bio_reverse_'+str(i), i2t_acc.item(), step)
                    #print(loss_image)
                    loss_clip_reverse += loss_image 
                    train_writer.add_scalar('loss/clip_bio_reverse_'+str(i), loss_clip_reverse.item(), step)      
                loss+=loss_clip_reverse

            if qwen_required:
                ortho_loss = orthogonality_loss(facial_probe, fine_probe)
                train_writer.add_scalar('loss/ortho_loss', ortho_loss.item(), step)
                loss+= ortho_loss_weight * ortho_loss
                
                '''
                # pair_loss1, pair_loss2 = contrastive_disentangle_loss(feat, f_bio_probe, f_non_bio_probe)
                # pair_loss1, pair_loss2 = contrastive_disentangle_loss(feat, facial_probe, fine_probe)
                # loss += pair_loss1
                # train_writer.add_scalar('loss/pair_loss1', pair_loss1.item(), step)
                # loss += pair_loss2
                # train_writer.add_scalar('loss/pair_loss2', pair_loss2.item(), step)
                '''
                
                # facial_probe_jepa = facial_probe.clone() ##chiranjeeev - 8 july
                # facial_probe_jepa = facial_probe_jepa.detach() ##chiranjeeev - 8 july
                jepa_loss = jepa_mse_loss(embed_probe_context_pred, embed_probe_target) ##chiranjeeev - 8 july
                
                # jepa_loss = jepa_infonce_loss(embed_probe_context_pred, facial_probe_jepa) ##chiranjeeev - 8 july
                train_writer.add_scalar('loss/jepa_loss', jepa_loss.item(), step)
                if torch.isnan(jepa_loss) or torch.isinf(jepa_loss):
                    print("Warning: jepa_loss is NaN, skipping this loss for the current step.")
                    jepa_loss = torch.tensor(0.0, device=feat[0].device)
                if is_jepa_threshold:
                    # print("is_jepa_threshold:", is_jepa_threshold)
                    loss+= jepa_loss_weight * jepa_loss
                
                llm_loss = 0.01 * llm_loss
                train_writer.add_scalar('loss/llm_loss', llm_loss.item(), step)
                if is_qwen_threshold:
                    # print("is_qwen_threshold:", is_qwen_threshold)
                    loss+= llm_loss
                    
                # print("loss:", loss.item(), "ID_LOSS:", ID_LOSS.item(), "TRI_LOSS:", TRI_LOSS.item(), "ortho_loss:", ortho_loss.item(), "pair_loss1:", pair_loss1.item(), "pair_loss2:", pair_loss2.item(), "llm_loss:", llm_loss.item(), "llm_loss_facial:", llm_loss_facial.item(), "llm_loss_fine:", llm_loss_fine.item())
                print("loss:", loss.item(), "ID_LOSS:", ID_LOSS.item(), "TRI_LOSS:", TRI_LOSS.item(), "ortho_loss:", ortho_loss.item(), "jepa_loss:", jepa_loss.item(), "llm_loss:", llm_loss.item(), "llm_loss_facial:", llm_loss_facial.item(), "llm_loss_fine:", llm_loss_fine.item())

            else:
                print("loss:", loss.item(), "ID_LOSS:", ID_LOSS.item(), "TRI_LOSS:", TRI_LOSS.item())
                
            return loss

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
            'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


