
import argparse
import copy
import json
import os
from itertools import product
import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2
import pandas as pd
from utils import *
from parameters import parser
from datasets.composition_dataset import CompositionDataset,MultiAttrCompositionDataset
from models.compositional_modules import get_model
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from torcheval.metrics import MultilabelAUPRC
cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"



class Evaluator_MAC:
    """
    Get seen/unseen/auc on MAC
    """

    def __init__(self, dset, model):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe',
        # 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [
            (dset.attr2idx[attr],
             dset.obj2idx[obj]) for attr,
            obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle
        # setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(
                topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        )
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        # Return only attributes that are in our pairs
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias  # Add bias to test pairs; add on unseen category

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k
        # largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        if len(pair_pred.shape) == 1:
            pair_pred = pair_pred.unsqueeze(1)
        results.update({'closed': (attr_pred, obj_pred,pair_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )
        train_pairs = self.dset.train_pairs

        #testset_pairs = self.dset.pairs
        testset_pairs = []
        for item in self.dset.data:
            pairs = [(a,item[2]) for a in item[1]]
            for p in pairs:
                if p not in testset_pairs:
                    testset_pairs.append(p)

        unseen_pairs = set(testset_pairs) - set(train_pairs)
        seen_pairs = set(testset_pairs) & set(train_pairs) # [('a','b')] str 

        seen_ind = []
        unseen_ind = []
        for idx,item in enumerate(self.dset.data):
            item_paris = [(a,item[2]) for a in item[1]]
            
            unseen_flag = False
            for p in item_paris:
                if p in unseen_pairs:
                    unseen_flag = True
            
            if unseen_flag:
                unseen_ind.append(idx)
            else:
                seen_ind.append(idx)

        print('seen_pairs',len(seen_pairs),len(seen_ind))
        print('unseen_pairs',len(unseen_pairs),len(unseen_ind))

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        mask = self.seen_mask.repeat(pair_truth.shape[0], 1)
        seen_pair_truth = pair_truth.clone()
        seen_pair_truth[~mask] = 0
        unseen_pair_truth = pair_truth.clone()
        unseen_pair_truth[mask] = 0


        
        def _process_multi_attr(_scores):

            # Top k accuracy for seen/unseen 

            pair_pred = _scores[2][:,:topk]
            seen_match = torch.gather(seen_pair_truth,1,pair_pred)
            seen_match = (seen_match.sum(dim=1) > 0).long().unsqueeze(1).float()   

            unseen_match = torch.gather(unseen_pair_truth,1,pair_pred)
            unseen_match = (unseen_match.sum(dim=1) > 0).long().unsqueeze(1).float()  
            # attr_match = torch.gather(attr_truth, 1, _scores[0][:, :topk])
            # attr_match  = (attr_match.sum(dim=1) > 0).long().unsqueeze(1)       


            seen_match = seen_match[seen_ind]
            unseen_match = unseen_match[unseen_ind]
            # Calculating class average accuracy

            return 0, 0, 0, seen_match, unseen_match, 0, 0, 0


        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            # attr_match = (
            #     attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            # )

            attr_match = torch.gather(attr_truth, 1, _scores[0][:, :topk])
            attr_match  = (attr_match.sum(dim=1) > 0).long().unsqueeze(1)     

            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            # Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        # correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
        #     unseen_ind
        # ]
        masked_scores = scores.masked_fill(pair_truth == 0, float('inf'))
        min_gt_scores = masked_scores.min(dim=1).values
        min_gt_scores = min_gt_scores[unseen_ind]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
            0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - min_gt_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process_multi_attr(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        # seen_accuracy.append(seen_match_max)
        # unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )


        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats



def test_czsl_MAC(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    """Function computes accuracy on MAC. (seen unseen auc)

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    

    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=config.bias, topk=config.topk
    )

    # attr_acc = float(torch.mean(
    #     (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    # obj_acc = float(torch.mean(
    #     (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=config.topk,
    )


    return stats




def clip_baseline(model, test_dataset, config, device):
    """Function to get the clip representations.

    Args:
        model (nn.Module): the clip model
        test_dataset (CompositionDataset): the test/validation dataset
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0

    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations with clip model.
    """
    if test_dataset.phase == "train":
        pairs_dataset = test_dataset.train_pairs
    else:
        pairs_dataset = test_dataset.pairs
    pairs = [(attr.replace(".", " ").lower(),
              obj.replace(".", " ").lower())
             for attr, obj in pairs_dataset]

    prompts = [f"a photo of {attr} {obj}" for attr, obj in pairs]

    tokenized_prompts = clip.tokenize(
        prompts, context_length= config.test_context_length)
    
    test_batch_tokens = np.array_split(
        tokenized_prompts,
        len(tokenized_prompts) //
        config.text_encoder_batch_size)


    #token_tensors = model.text_encoder.token_embedding(tokenized_prompts.cuda()).type(model.dtype)
    rep = torch.Tensor().to(device).type(model.dtype)
    

    with torch.no_grad():
        for batch_tokens in test_batch_tokens:
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(
                batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            rep = torch.cat((rep, text_features), dim=0)

    return rep

def clip_primitive_text_rep(model, test_dataset, config, device):
    """Function to get the clip representations.

    Args:
        model (nn.Module): the clip model
        test_dataset (CompositionDataset): the test/validation dataset
        config (argparse.ArgumentParser): config/args
        device (str): device type cpu/cuda:0

    Returns:
        torch.Tensor: returns the tensor with the attribute-object
            representations with clip model.
    """
    
    allattrs = train_dataset.attrs
    allobjs = train_dataset.objs

    # cleaning the classes and the attributes
    allobjs = [cla.replace(".", " ").lower() for cla in allobjs]
    allattrs = [attr.replace(".", " ").lower() for attr in allattrs]

    prompts = [f"a photo of {p}" for p in allattrs + allobjs]

    tokenized_prompts = clip.tokenize(
        prompts, context_length= config.test_context_length)
    
    
    test_batch_tokens = np.array_split(
        tokenized_prompts,
        len(tokenized_prompts) //
        64)


    #token_tensors = model.text_encoder.token_embedding(tokenized_prompts.cuda()).type(model.dtype)
    primitive_rep = torch.Tensor().to(device).type(model.dtype)
    

    with torch.no_grad():
        for batch_tokens in test_batch_tokens:
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(
                batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            primitive_rep = torch.cat((primitive_rep, text_features), dim=0)

    return primitive_rep




def predict_logits(model, dataset, config,text_rep=None):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    # print(text_rep.shape)
    if dataset.phase == "train":
        pairs_dataset = dataset.train_pairs
    else:
        pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    all_predict = torch.Tensor()

    loss = 0
    progress_bar = tqdm(total=len(dataloader), desc="Testing")
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            
            # batch_img = data[0].cuda()
            predict = model(data, pairs)
            # save origin predict

            logits = model.logit_infer_multi_attr(predict, pairs)
            loss += model.multi_attr_loss_calu(predict, data).item()

            if isinstance(predict,list) or isinstance(predict,tuple):
                predict = torch.cat(predict,dim=1).cpu()
            all_predict = torch.cat([all_predict,predict.cpu()],dim=0)

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

            if (idx+1) % 10 == 0:
                progress_bar.update(50)

    # attr/pair -> multi hot vector;   obj_gt -> index
    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits,all_predict ,all_attr_gt, all_obj_gt, all_pair_gt, loss / len(dataloader)



def predict_logits_clip(model,dataset,config,text_rep=None):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations.

    Args:
        model (nn.Module): the model
        dataset (CompositionDataset): the composition dataset (validation/test)
        config (argparse.ArgumentParser): config/args
        text_rep : text representations of CLIP
    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_logits = torch.Tensor()

    loss = 0


    if test_dataset.phase == "train":
        pairs_dataset = test_dataset.train_pairs
    else:
        pairs_dataset = test_dataset.pairs

    pairs = [(attr.replace(".", " ").lower(),
              obj.replace(".", " ").lower())
             for attr, obj in pairs_dataset]

    prompts = [f"a photo of {attr} {obj}" for attr, obj in pairs]

    tokenized_prompts = clip.tokenize(
        prompts, context_length= config.test_context_length)
    
    test_batch_tokens = np.array_split(
        tokenized_prompts,
        len(tokenized_prompts) //
        config.text_encoder_batch_size)


    # token_tensors = model.text_encoder.token_embedding(tokenized_prompts.cuda()).type(model.dtype)

    with torch.no_grad():
        for batch_tokens in test_batch_tokens:
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(
                batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            rep = torch.cat((rep, text_features), dim=0)


    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            #  text_tokens = tokenized_prompts.to(device)

            # text_rep = model.text_encoder(text_tokens, enable_pos_emb=True)
            # text_rep = text_rep / text_rep.norm(
            #                 dim=-1, keepdim=True
            #             )

            batch_img = data[0].to(device)
            batch_img_feat = model.encode_image(batch_img)
            normalized_img = batch_img_feat / batch_img_feat.norm(
                dim=-1, keepdim=True
            )

            logits = (
                model.clip_model.logit_scale.exp()
                * normalized_img
                @ text_rep.t()
            )

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)

            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, None , all_attr_gt, all_obj_gt, all_pair_gt,loss



def predict_logits_open_world(model, dataset, config,text_rep=None):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    # print(text_rep.shape)
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    loss = 0

    text_eval_batch = 1024
    with torch.no_grad():

        text_feats = model.encode_text_for_open(pairs)

        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            # batch_img = data[0].cuda()
            predict = model.forward_for_open(data, pairs, text_feats)

            logits = model.logit_infer_multi_attr(predict, pairs)
            loss += model.multi_attr_loss_calu(predict, data).item()
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    # ? delete the text encoder to save CUDA memory
    # del model.transformer
    # torch.cuda.empty_cache()

    return all_logits,None, all_attr_gt, all_obj_gt, all_pair_gt, loss / len(dataloader)

def compute_F1(predictions, labels):
    '''
    Args:
        predictions: multi_hot 1-0 tensor; process the logits with threshlod/topK first
        labels: multi_hot 1-0 tensor
    '''
    # idx = predictions.topk(dim=1, k=k_val)[1]
    # predictions.fill_(0)
    # predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val).to(predictions.device))
    mask = predictions == 1
    TP = (labels[mask] == 1).sum().float()
    tpfp = mask.sum().float()
    tpfn = (labels == 1).sum().float()
    p = TP / tpfp
    r = TP/tpfn
    f1 = 2*p*r/(p+r)

    return f1.item(), p.item(), r.item()

def cumpute_exactP(predictions,labels):
    '''
    Args:
        predictions: multi_hot 1-0 tensor; process the logits with threshlod/topK first
        labels: multi_hot 1-0 tensor
    '''
    exact_equal = torch.all(torch.eq(predictions,labels),dim=1)

    return exact_equal

# the mimimal num to cover all ground truth
def compute_coverage(predictions,labels):
    # get the sort index of all categories
    sorted_idx =  torch.argsort(predictions, dim=1,descending=True)

    rank_tensor = torch.empty_like(predictions, dtype=torch.long)
    rank_tensor.scatter_(1, sorted_idx, torch.arange(predictions.size(1)).repeat(predictions.shape[0],1))
    # get the index of gt
    
    coverage_idx = rank_tensor * labels
    # get the mimimal num to cover all gt
    coverage_steps = torch.max(coverage_idx,dim=1).values.float() + 1

    coverage = (coverage_steps - 1).float().mean()

    label_nums = torch.sum(labels,dim=1)

    normalized_coverage = torch.div(1, torch.exp(coverage_steps/label_nums -1) ).mean()
    

    return coverage.item(),normalized_coverage.item()

def getUnbiasedPrecition(
        test_dataset,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config,
        save_exact_match = False):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    phase_pairs = []
    predictions = all_logits
    train_pairs = test_dataset.train_pairs
    all_pairs = test_dataset.pairs
    device = all_logits.device

    
    if test_dataset.phase == 'val':
        phase_pairs = test_dataset.val_pairs
    elif test_dataset.phase == 'test':
        phase_pairs = test_dataset.test_pairs
    elif test_dataset.phase == 'train':
        phase_pairs = test_dataset.train_pairs
        all_pairs = test_dataset.train_pairs

    pairs_num = len(all_pairs)

    pairs2idx = torch.tensor([(test_dataset.attr2idx[attr], test_dataset.obj2idx[obj])
                            for attr, obj in all_pairs]).to(device)

    # convert obj gt to one hot vector
    all_obj_gt = torch.nn.functional.one_hot(all_obj_gt, num_classes=len(test_dataset.objs))

    if all_pair_gt.shape[-1] != pairs_num:
        all_pair_gt = torch.zeros((all_pair_gt.shape[0],pairs_num), dtype=torch.int).scatter_(1,all_pair_gt,1)

    
    seen_id,unseen_id,exsiting_id = [],[],[]
    for idx,pair in enumerate(all_pairs):
        if pair in phase_pairs:
            exsiting_id.append(idx)

        if pair in train_pairs:
            seen_id.append(idx)
        else:
            unseen_id.append(idx)
    
    seen_id,unseen_id,exsiting_id = torch.LongTensor(seen_id), torch.LongTensor(unseen_id), torch.LongTensor(exsiting_id)


    # precision for obj/attr
    primitive_topk = 1
    _, predict_idx = all_logits.topk(dim=1,k=primitive_topk)
    predict_idx = predict_idx.contiguous().view(-1)
    obj_predict_idx = pairs2idx[predict_idx][:,1].view(-1,primitive_topk)
    attr_predict_idx = pairs2idx[predict_idx][:,0].view(-1,primitive_topk)

    top1_obj_predictions = torch.zeros([predictions.size(0), all_obj_gt.shape[-1]])
    top1_obj_predictions.scatter_(dim=1, index=obj_predict_idx, value=1).to(all_obj_gt.device)
    _,obj_p,_ = compute_F1(top1_obj_predictions, all_obj_gt)

    top1_attr_predictions = torch.zeros([predictions.size(0), all_attr_gt.shape[-1]])
    top1_attr_predictions.scatter_(dim=1, index=attr_predict_idx, value=1).to(all_attr_gt.device)
    _,attr_p,_ = compute_F1(top1_attr_predictions, all_attr_gt)

    # compute the top1 accuracy
    idx = predictions.topk(dim=1, k=1)[1]
    top1_predictions = torch.zeros([predictions.size(0),pairs_num])
    top1_predictions.scatter_(dim=1, index=idx, value=1).to(predictions.device)
    _,top1_pair_p,_ = compute_F1(top1_predictions, all_pair_gt)

    # compute the topK recall
    idx = predictions.topk(dim=1, k=3)[1]
    top1_predictions = torch.zeros([predictions.size(0),pairs_num])
    top1_predictions.scatter_(dim=1, index=idx, value=1).to(predictions.device)
    _,_,top3_pair_r = compute_F1(top1_predictions, all_pair_gt)


    # compute the topK recall
    idx = predictions.topk(dim=1, k=5)[1]
    top1_predictions = torch.zeros([predictions.size(0),pairs_num])
    top1_predictions.scatter_(dim=1, index=idx, value=1).to(predictions.device)
    _,_,top5_pair_r = compute_F1(top1_predictions, all_pair_gt)


    # compte the exact match; select topk based on the ground truth
    gt_num = all_pair_gt.sum(dim=-1,keepdim=False)
    gt_num_predicitons = torch.zeros([predictions.size(0),pairs_num])  

    for idx,num in enumerate(gt_num):  
        _, indices = predictions[idx].topk(int(num), dim=0)  
        
        gt_num_predicitons[idx, indices] = 1  
    exact_match = cumpute_exactP(gt_num_predicitons, all_pair_gt)


    exact_p =  exact_match.float().mean().item()

    if save_exact_match == True:
        save_path = os.path.join('./result',config.experiment_name)
        os.makedirs(save_path, exist_ok=True)

        exact_match = exact_match.tolist()

        save_file = os.path.join(save_path, f'exact_match_{test_dataset.phase}_{config.open_world}.json')
        with open(save_file, 'w') as f:
            json.dump(exact_match, f)



    # compute the map for exsiting pair in test/val split
    ap_metric = MultilabelAUPRC(num_labels=all_logits.shape[1], average=None)
    ap_metric.update(all_logits,all_pair_gt)
    mAP = ap_metric.compute()
    mAP = mAP[exsiting_id].mean().item()

    # compute the coverage:
    coverage,normalized_coverage = compute_coverage(predictions,all_pair_gt)


    # compute the topK 


    results = {'top1_P':top1_pair_p,
               'top3_R':top3_pair_r, 
               'top5_R':top5_pair_r,
              'Exact_p':exact_p,
              'mAP':mAP,
              'coverage':coverage,
              'normailized_coverage': normalized_coverage,
              'obj_p':obj_p,
              'attr_p':attr_p,
              'coverage_K':coverage
              }
    

    return results


def test_CZSL(model, testloader, evaluator,  args, threshold=None, print_results=True):

        model.eval()
        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
            data = [d.to(device) for d in data]

            _, predictions = model(data)

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

            all_pred.append(predictions)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

        if args.cpu_eval:
            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
        else:
            all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
                'cpu'), torch.cat(all_pair_gt).to('cpu')

        # Calculate best unseen accuracy
        all_pred_dict = torch.cat(all_pred, dim=0)
        results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
        stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict,
                                               topk=args.topk)

        result = ''
        for key in stats:
            result = result + key + '  ' + str(round(stats[key], 4)) + '| '

        result = result + args.name
        if print_results:
            print(f'Results')
            print(result)
        return results

if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)


        
    if config.debug:
        use_wandb = False

    # config.open_world = True
    # set the seed value
    print("----")
    #config.open_world = True
    test_type = 'OPEN WORLD' if config.open_world else 'CLOSED WORLD'
    print(f"{test_type} evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")

    
    dataset_path = config.dataset_path
    config.context_length = config.test_context_length



    print('loading validation dataset')
    # val_dataset = CompositionDataset(dataset_path,
    #                                  phase='val',
    #                                  split='compositional-split-natural',
    #                                  open_world=config.open_world)
    # config.split = 'sample_complex_all_data'
    split = config.split

    print(split)
    
    

    train_dataset = MultiAttrCompositionDataset(root=dataset_path,
                                            phase='train',
                                            split=split,
                                            open_world=config.open_world)

    # val_dataset = MultiAttrCompositionDataset(root=dataset_path,
    #                                             phase='val',
    #                                             split=split,
    #                                             open_world=config.open_world)

    test_dataset = MultiAttrCompositionDataset(root=dataset_path,
                                            phase='test',
                                            split=split,
                                            open_world=config.open_world)


    #  test_dataset.data = test_dataset.data[:100]

    predict_logits_func = predict_logits

    val_text_rep = None
    test_text_rep = None
    train_text_rep = None


    
    if config.load_model :
        print("load model from:",config.load_model)

    if config.model_name == 'clip':
        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.test_context_length)

        model = CLIPInterface(
            clip_model,
            config,
            token_ids=None,
            device=device,
            enable_pos_emb=True)
        model.eval()

        primitive_text_rep = clip_primitive_text_rep(model, train_dataset, config, device)

        # train_text_rep = clip_baseline(model, train_dataset, config, device)
        # val_text_rep = clip_baseline(model, val_dataset, config, device)
        test_text_rep = clip_baseline(model, test_dataset, config, device)
        test_text_rep = None

        predict_logits_func = predict_logits_clip
    else:
        model,optimizer = get_model(train_dataset,config,device)
        model = model.cuda()
        model.eval()


        if config.load_model:
            model.load_state_dict(torch.load(config.load_model))
    
    if config.open_world == True:
        predict_logits_func = predict_logits_open_world


    val_stats = {}
      
    # test_dataset = train_dataset
    print('evaluating on the test set')
    with torch.no_grad():
        evaluator = Evaluator_MAC(test_dataset, model=None)
        all_logits,all_predict , all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits_func(
            model, test_dataset, config,test_text_rep)

        test_stats = getUnbiasedPrecition(
            test_dataset,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config,
            save_exact_match=True
        )

        print(test_stats)
        evaluator = Evaluator_MAC(test_dataset, model=None)

        test_stats_czsl = test_czsl_MAC(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )

        test_stats['seen'] = test_stats_czsl['best_seen']
        test_stats['unseen'] = test_stats_czsl['best_unseen']
        test_stats['AUC'] = test_stats_czsl['AUC']

        evaluate_dict = {'logits':all_logits,'origin_logits':all_predict,'attr_gt':all_attr_gt,'obj_gt':all_obj_gt,'pair_gt':all_pair_gt}

        #torch.save(evaluate_dict,f'./result/{config.experiment_name}_{config.open_world}.pt')

        result = ""


        for key in test_stats:
            result = result + key + "  " + \
                str(round(test_stats[key], 4)) + "| "
        print(result)

    results = {
        'val': val_stats,
        'test': test_stats,
    }

    # save results
    final_keys = ['top1_P','top5_R','Exact_p','coverage','obj_p','attr_p','seen','unseen','AUC']

    
    # 定义目标 CSV 文件路径
    csv_path = './result/results_summary.csv'

    record = {key: round(results['test'].get(key, float('nan')), 4) * (1.0 if key == 'coverage' else 100.0) for key in final_keys}
    
    
    record['experiment_name'] = config.experiment_name
    record['model'] = config.load_model.split('/')[-1]
    record['setting'] = config.open_world

    new_row = pd.DataFrame([record])
    unique_keys = ['experiment_name', 'model', 'setting']

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        match_idx = df[
            (df['experiment_name'] == record['experiment_name']) &
            (df['model'] == record['model']) &
            (df['setting'] == record['setting'])
        ].index

        if len(match_idx) > 0:
            df.loc[match_idx[0]] = record
        else:
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # 将 experiment_name, model, setting 放在前面
    priority_cols = ['experiment_name', 'model', 'setting']
    other_cols = [col for col in df.columns if col not in priority_cols]
    df = df[priority_cols + other_cols]

    df.to_csv(csv_path, index=False)


    if config.load_model:
        title = config.load_model[:-2]
    else:
        os.makedirs(config.save_path, exist_ok=True)
        title = config.save_path + '/'

    if config.open_world:
        result_path = title + "open.calibrated.json"
    else:
        result_path = title + "closed.json"

    with open(result_path, 'w+') as fp:
        json.dump(results, fp)

    print("done!")
