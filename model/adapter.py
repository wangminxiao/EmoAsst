from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from data.tokenizer import tokenize

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, clip_model, device):
    template = ['some one is speaking with a {} emotion.']
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            texts = [t.format(classname) for t in template]
            texts = tokenize(texts).to(device)

            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache, device):
    clip_model.eval()
    if cfg['load_cache'] == True:
        try:
            cache_keys_a = torch.load(cfg['cache_dir'] + '/keys_a_' + str(cfg['shots']) + "shots.pt")
            cache_keys_v = torch.load(cfg['cache_dir'] + '/keys_v_' + str(cfg['shots']) + "shots.pt")
            cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
        except:
            cfg['load_cache'] = False

    if cfg['load_cache'] == False:
        cache_keys_a = []
        cache_keys_v = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features_v = []
                train_features_a = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, batch in enumerate(tqdm(train_loader_cache)):
                    videos, texts_sentiment_logits  = batch[0], batch[1]
                    texts = texts_sentiment_logits['utt_token']
                    audio = texts_sentiment_logits['audio_wav']
                    target = texts_sentiment_logits['emotion_idx']

                    videos = videos.to(device=device)
                    texts = texts.to(device=device)
                    audio = audio.to(device=device)
                    target = target.to(device=device, non_blocking=True)

                    video_features, _, audio_features, _ = clip_model(videos, texts, audio)
                    train_features_v.append(video_features)
                    train_features_a.append(audio_features)

                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)

                cache_keys_a.append(torch.cat(train_features_a, dim=0).unsqueeze(0))
                cache_keys_v.append(torch.cat(train_features_v, dim=0).unsqueeze(0))

        cache_keys_a = torch.cat(cache_keys_a, dim=0).mean(dim=0)
        cache_keys_a /= cache_keys_a.norm(dim=-1, keepdim=True)
        cache_keys_a = cache_keys_a.permute(1, 0)

        cache_keys_v = torch.cat(cache_keys_v, dim=0).mean(dim=0)
        cache_keys_v /= cache_keys_v.norm(dim=-1, keepdim=True)
        cache_keys_v = cache_keys_v.permute(1, 0)

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys_a, cfg['cache_dir'] + '/keys_a_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_keys_v, cfg['cache_dir'] + '/keys_v_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return (cache_keys_a, cache_keys_v), cache_values


def pre_load_features(cfg, split, clip_model, loader, device):
    clip_model.eval()
    if cfg['load_pre_feat'] == True:
        try:
            features_a = torch.load(cfg['cache_dir'] + "/" + split + "_a_f.pt")
            features_v = torch.load(cfg['cache_dir'] + "/" + split + "_v_f.pt")
            labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
        except:
            cfg['load_pre_feat'] = False
    
    if cfg['load_pre_feat'] == False:
        features_a, features_v, labels = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader)):
                videos, texts_sentiment_logits = batch[0], batch[1]
                texts = texts_sentiment_logits['utt_token']
                audio = texts_sentiment_logits['audio_wav']
                target = texts_sentiment_logits['emotion_idx']

                videos = videos.to(device=device)
                texts = texts.to(device=device)
                audio = audio.to(device=device)
                target = target.to(device=device, non_blocking=True)

                video_features, _, audio_features, _ = clip_model(videos, texts, audio)
                features_a.append(audio_features)
                features_v.append(video_features)
                labels.append(target)

        features_a, features_v, labels = torch.cat(features_a), torch.cat(features_v), torch.cat(labels)

        torch.save(features_a, cfg['cache_dir'] + "/" + split + "_a_f.pt")
        torch.save(features_v, cfg['cache_dir'] + "/" + split + "_v_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    return (features_a, features_v), labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    adapter_a, adapter_v = adapter
                    audio_features, video_features = features
                    affinity = 0.5 * (adapter_a(audio_features) + adapter_v(video_features))#
                else:
                    cache_keys_a, cache_keys_v = cache_keys
                    audio_features, video_features = features
                    affinity = 0.5 * (audio_features @ cache_keys_a + video_features @ cache_keys_v)

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                audio_features, video_features = features
                clip_logits = 100. * (audio_features @ clip_weights + video_features @ clip_weights) / 2.
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set. --------")
    cache_values = cache_values.float()
    cache_keys_a, cache_keys_v = cache_keys
    val_features_a, val_features_v = val_features
    test_features_a, test_features_v = test_features

    # Zero-shot CLIP
    clip_logits = 100. * (val_features_a @ clip_weights + val_features_v @ clip_weights) / 2.
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = val_features_a @ cache_keys_a + val_features_v @ cache_keys_v
    print(affinity.type(), cache_values.type())
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * (test_features_a @ clip_weights + test_features_v @ clip_weights) / 2.
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features_a @ cache_keys_a + test_features_v @ cache_keys_v
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F, device):
    cache_values = cache_values.float()
    cache_keys_a, cache_keys_v = cache_keys
    test_features_a, test_features_v = test_features
    # Enable the cached keys to be learnable
    clip_model.train()
    adapter_a = nn.Linear(cache_keys_a.shape[0], cache_keys_a.shape[1], bias=False).cuda()
    adapter_a.weight = nn.Parameter(cache_keys_a.t())

    adapter_v = nn.Linear(cache_keys_v.shape[0], cache_keys_v.shape[1], bias=False).cuda()
    adapter_v.weight = nn.Parameter(cache_keys_v.t())

    optimizer = torch.optim.AdamW([
                                   {'params': adapter_a.parameters(), 'lr': cfg['lr']},
                                   {'params': adapter_v.parameters(), 'lr': cfg['lr']},
                                   ],
                                  lr=cfg['lr'], eps=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter_a.train()
        adapter_v.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        for i, batch in enumerate(tqdm(train_loader_F)):
            videos, texts_sentiment_logits = batch[0], batch[1]
            texts = texts_sentiment_logits['utt_token']
            audio = texts_sentiment_logits['audio_wav']
            target = texts_sentiment_logits['emotion_idx']

            videos = videos.to(device=device)  #
            texts = texts.to(device=device)
            audio = audio.to(device=device)
            target = target.to(device=device, non_blocking=True)
            with torch.no_grad():
                video_features, _, audio_features, _ = clip_model(videos, texts, audio)
                video_features /= video_features.norm(dim=-1, keepdim=True)
                audio_features /= audio_features.norm(dim=-1, keepdim=True)

            affinity = 0.5 * (adapter_a(audio_features) + adapter_v(video_features))#
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * (audio_features @ clip_weights + video_features @ clip_weights) / 2.
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        current_lr = cfg['lr']#scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter_a.eval()
        adapter_v.eval()

        affinity =  0.5 * (adapter_a(test_features_a) + adapter_v(test_features_v))
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * (test_features_a @ clip_weights + test_features_v @ clip_weights) / 2.
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter_a.weight, cfg['cache_dir'] + "/best_F_a_" + str(cfg['shots']) + "shots.pt")
            torch.save(adapter_v.weight, cfg['cache_dir'] + "/best_F_v_" + str(cfg['shots']) + "shots.pt")

    adapter_a.weight = torch.load(cfg['cache_dir'] + "/best_F_a_" + str(cfg['shots']) + "shots.pt")
    adapter_v.weight = torch.load(cfg['cache_dir'] + "/best_F_v_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights,
                                      adapter=(adapter_a, adapter_v))

    print("\n-------- Evaluating on the test set. --------")

    affinity = 0.5 * (adapter_a(test_features_a) + adapter_v(test_features_v))
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


