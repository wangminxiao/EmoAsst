import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MultiPosConLossMM(nn.Module):
    """Multi-positive contrastive loss, when multiple images corresponds to the same texts"""
    def __init__(self, temperature=0.1, w1=1.0, w2=1.0, distr=False):
        """
        Args:
            temperature: temperature for contrastive loss
            w1: weight for the image contrastive part
            w2: weight for the image-text part
        """
        super(MultiPosConLossMM, self).__init__()
        self.temperature = temperature
        self.w1 = w1
        self.w2 = w2
        self.last_local_batch_size = None
        self.v_label_matrix = None
        self.t_label_matrix = None
        self.mask = None
        self.logits_mask = None
        self.distr = distr


    def forward(self, outputs):
        v_feats = outputs['image_emb']
        t_feats = outputs['text_emb']
        a_feats = outputs['audio_emb']
        v_labels = outputs['gt']
        t_labels = outputs['gt']
        a_labels = outputs['gt']
        logit_scale = outputs['logit_scale']
        device = (torch.device('cuda')
                  if v_feats.is_cuda
                  else torch.device('cpu'))

        v_feats = F.normalize(v_feats, dim=-1, p=2)
        t_feats = F.normalize(t_feats, dim=-1, p=2)
        a_feats = F.normalize(a_feats, dim=-1, p=2)

        v_local_batch_size = v_feats.size(0)
        t_local_batch_size = t_feats.size(0)
        a_local_batch_size = a_feats.size(0)

        if self.distr:
            all_v_feats = torch.cat(torch.distributed.nn.all_gather(v_feats), dim=0)
            all_t_feats = torch.cat(torch.distributed.nn.all_gather(t_feats), dim=0)
            all_a_feats = torch.cat(torch.distributed.nn.all_gather(a_feats), dim=0)
        else:
            all_v_feats = v_feats
            all_t_feats = t_feats
            all_a_feats = a_feats

        # compute the logits for image-text contrasting
        logits_v = logit_scale * torch.matmul(v_feats, all_t_feats.T)
        logits_tv = logit_scale * torch.matmul(t_feats, all_v_feats.T)
        logits_a = logit_scale * torch.matmul(a_feats, all_t_feats.T)
        logits_ta = logit_scale * torch.matmul(t_feats, all_a_feats.T)

        # compute the logits for image-only contrasting
        feats = outputs['text_emb']
        feats = F.normalize(feats, dim=-1, p=2)
        if self.distr:
            all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        else:
            all_feats = feats
        logits = logit_scale * torch.matmul(feats, all_feats.T) #/ self.temperature

        # 
        if self.distr:
            all_v_labels = concat_all_gather(v_labels)
            all_t_labels = concat_all_gather(t_labels)
            all_a_labels = concat_all_gather(a_labels)
        else:
            all_v_labels = v_labels
            all_t_labels = t_labels
            all_a_labels = a_labels
        all_v_labels = all_v_labels.contiguous().view(1, -1)
        all_t_labels = all_t_labels.contiguous().view(1, -1)
        all_a_labels = all_a_labels.contiguous().view(1, -1)

        # mask matrix for image-text contrastive loss
        self.v_label_matrix = torch.eq(v_labels.view(-1, 1),
                                        all_t_labels).float().to(device)
        self.t_label_matrix = torch.eq(t_labels.view(-1, 1),
                                        all_v_labels).float().to(device)
        self.a_label_matrix = torch.eq(a_labels.view(-1, 1),
                                        all_t_labels).float().to(device)

        # mask matrix for image supervised contrastive loss
        self.mask = torch.eq(v_labels.view(-1, 1), all_v_labels).float().to(device)

        # image only loss
        mask = self.mask
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        img_loss = compute_cross_entropy(p, logits)

        # image text loss
        v_mask = self.v_label_matrix
        p_v = v_mask / v_mask.sum(1, keepdim=True).clamp(min=1.0)
        t_mask = self.t_label_matrix
        p_t = t_mask / t_mask.sum(1, keepdim=True).clamp(min=1.0)
        a_mask = self.a_label_matrix
        p_a = a_mask / a_mask.sum(1, keepdim=True).clamp(min=1.0)

        img_txt_loss = compute_cross_entropy(p_v, logits_v) + (compute_cross_entropy(p_t, logits_tv) / 2)
        audio_txt_loss = compute_cross_entropy(p_a, logits_a) + (compute_cross_entropy(p_t, logits_ta) / 2)

        # total loss
        loss = self.w1 * img_loss + self.w2 * img_txt_loss + self.w2 * audio_txt_loss

        return {'loss': loss,
                'image_loss': img_loss,
                'img_txt_loss': img_txt_loss}