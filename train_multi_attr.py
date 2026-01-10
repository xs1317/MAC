import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['WANDB_MODE'] = 'offline'
#os.environ['WANDB_BASE_URL'] = "https://api.bandw.top"
import pickle
import wandb
import numpy as np
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from models.compositional_modules import get_model
from parameters import parser
from collections import OrderedDict

from test_multi_attr import Evaluator_MAC, test_czsl_MAC
import test_multi_attr as test
from datasets.composition_dataset import MultiAttrCompositionDataset
from utils.core import *


class WandbWriter:
    def __init__(self, exp_name, cfg, output_dir, cur_step=0, step_interval=0):
        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval
        wandb.init(project="CZSL", name=exp_name, config=cfg, dir=output_dir)

    def write_log(self, stats: OrderedDict , step=0):
        self.step += 1
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue

            log_dict = {}
            for var_name, val in loader_stats.items():
                log_dict.update({loader_name + '/' + var_name: val})

            self.wandb.log(log_dict, step=step)


def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset,wandb_logger=None):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    update_step = 30
    global_step = 0
    if wandb_logger!=None:
        wandb_logger.step = update_step

    model.train()
    
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    # train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
    #                             for attr, obj in train_dataset.train_pairs]).cuda()


    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                 for attr, obj in train_dataset.train_pairs]).cuda()
    
    
    if hasattr(config,'positive_weight') and config.positive_weight == False:
        print('Do not use positive class weight for BCE')
        pair_positive_weight = None
        attr_positive_weight = None
    else:
        pair_positive_weight = torch.div(len(train_dataset),torch.Tensor(train_dataset.ao_pair_num)).cuda()
        attr_positive_weight = torch.div(len(train_dataset),torch.Tensor(train_dataset.a_num)).cuda()


    train_losses = []
   # all_evaluate_dict = []

    train_set_evaluate_result = {}
    test_set_evaluate_result = {}
    val_set_evaluate_result = {}
    
    if hasattr(config, 'evaL_epoch'):
        print('eval every epoch',config.eval_epoch)

    for i in range(config.epoch_start, config.epochs):
        torch.cuda.empty_cache()
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )
        model.train()

        epoch_train_losses = []

        predict_logits_list = []

        for bid, batch in enumerate(train_dataloader):
            global_step += 1    

            # release the cache
            

            model.train()
            predict = model(batch, train_pairs)

            # 处理 predict 是 list 或 tensor 的情况
            if isinstance(predict, list) or isinstance(predict, tuple):
                predict_logits_list.append([p.detach().cpu() for p in predict])
            else:
                predict_logits_list.append(predict.detach().cpu())

            loss = model.multi_attr_loss_calu(predict, batch,pair_positive_weight,attr_positive_weight)
            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps
            # backward pass
            loss.backward()

            # cut grad
            for param_group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=2.0)
                
            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))
            epoch_train_losses.append(loss.item())

            if (bid+1) % update_step  == 0:
                progress_bar.set_postfix(
                    {"train loss": np.mean(epoch_train_losses[-update_step:])}
                )

                progress_bar.update(update_step )

                if wandb_writer !=None:
                    wandb.log({"train/loss":np.mean(epoch_train_losses[-update_step:])},
                            step = global_step )


        # save origin logits for analysis   
        # sample shuffle; meaningless
        # save_path = f'./result/{config.experiment_name}'   
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        

        # if (i + 1) % config.save_every_n == 0 or i ==0:
        #     logits_path = os.path.join(save_path, f"epoch_{i}_logits.pt")
        #     torch.save(predict_logits_list, logits_path)

        progress_bar.close()
        progress_bar.write(f"epoch {i+1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        # save params and optimizer state
        if (i + 1) % config.save_every_n == 0:
            model.save_params(config,i)
            # torch.save(optimizer.state_dict(), os.path.join(config.save_path, f"optimizer_epoch_{i}.pt"))

        wandb_log_dict = {}
        evaluate_dict = {}
        train_result= {'epoch':i}

        config.evalute_train_set = False
        if config.evalute_train_set == True:
            print("Evaluating train dataset:")
            train_result, train_evaluate_dict = evaluate(model, train_dataset, config)
            train_result.update(train_result)
            evaluate_dict['train'] = train_evaluate_dict
            wandb_log_dict['train'] = train_result
            
            train_set_evaluate_result[i]=train_result
            with open(f"{config.save_path}/train_evaluate.json",'w') as f:
                json.dump(train_set_evaluate_result,f,indent=4)

        evaluate_val_set = False
        if evaluate_val_set:
            print("Evaluating val dataset:")
            val_result,val_evaluate_dict = evaluate(model, val_dataset, config)
            val_result.update(val_result)
            evaluate_dict['val'] = val_evaluate_dict 
            wandb_log_dict['val'] = val_result
            wandb_log_dict['val']['epoch'] = i
            
            val_set_evaluate_result[i]=val_result
            with open(f"{config.save_path}/val_evaluate.json",'w') as f:
                json.dump(val_set_evaluate_result,f,indent=4)

        # eval on test set every 5 epoch
        eval_epoch = 5
        if hasattr(config, 'eval_epoch'):
            eval_epoch = config.eval_epoch 
        if config.evalute_test_set == True and  ((i + 1) % eval_epoch == 0):
            print("Evaluating test dataset:") 
            test_result,test_evaluate_dict = evaluate(model, test_dataset, config)
            evaluate_dict['test'] = test_evaluate_dict
            wandb_log_dict['test'] = test_result
            wandb_log_dict['test']['epoch'] = i

            test_set_evaluate_result[i]=test_result
            with open(f"{config.save_path}/test_evaluate.json",'w') as f:
                json.dump(test_set_evaluate_result,f,indent=4)

        
        if wandb_logger!=None:
            wandb_logger.write_log(
                wandb_log_dict,
                step=global_step
            )
    
    
    #torch.save(all_evaluate_dict,os.path.join(config.save_path,'train_predict_logits.pt'))
    if wandb_logger!=None:
        wandb.finish()
    if config.save_final_model:
        model.save_params(config,i,"final_model.pt")
        # torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def get_attr_result(
        all_attr_gt,
        all_attr_logits):
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
    predictions = all_attr_logits


    # 计算某个属性的准确度top-3
    attr_topk = 3
    _, attr_logits_topk_idx = all_attr_logits.topk(dim=1,k=attr_topk)
    topk_attr_predictions = torch.zeros([predictions.size(0), all_attr_gt.shape[-1]])
    topk_attr_predictions.scatter_(dim=1, index=attr_logits_topk_idx, value=1).to(all_attr_gt.device)


    predicted = topk_attr_predictions.sum(dim=0)
    correct = (topk_attr_predictions * all_attr_gt).sum(dim=0) 

    # 处理分母为零的情况，避免除以零
    recall = torch.where(
        all_attr_gt.sum(dim=0) > 0,
        correct.float() / all_attr_gt.sum(dim=0).float(),
        torch.zeros_like(correct, dtype=torch.float)
    )


    precision = torch.where(
        predicted > 0,
        correct.float() / predicted.float(),
        torch.zeros_like(correct, dtype=torch.float)
    )

    return {f'r':recall, f'p':precision}

def evaluate(model, dataset, config):

    model.eval()
    all_logits,all_predict,all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    

    test_stats = test.getUnbiasedPrecition(
        dataset,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    
    # get result for seen/unseen/auc
    evaluator = Evaluator_MAC(dataset, model=None)

    test_stats_czsl = test_czsl_MAC(
        dataset,
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

    # interest_attributes = ['rectangular']

    # interest_attributes_idx = [test_dataset.attrs.index(a) for a in interest_attributes]
    # results = get_attr_result(all_attr_gt,all_predict[:,:len(dataset.attrs)])
    
    # idx = interest_attributes_idx[0]
    # recall,p = results['r'],results['p']


    # test_stats['rectangular_r'] = recall[idx].item()
    # test_stats['rectangular_p'] = p[idx].item()

    test_saved_results = dict()
    result = ""
    key_set = test_stats.keys()
    for key in key_set:
        if key in test_stats.keys():
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
            test_saved_results[key] = round(test_stats[key], 4)
    print(result)


    test_saved_results['best_loss'] = loss_avg

    evaluate_dict = {'logits':all_logits.cpu(),'origin_logits':all_predict.cpu(),'attr_gt':all_attr_gt.cpu(),'obj_gt':all_obj_gt.cpu(),'pair_gt':all_pair_gt.cpu()}
    return test_saved_results,  evaluate_dict



if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)


    set_seed(config.seed)
    local_rank = 0

    use_wandb = config.wandb

    # use_wandb = False

    if config.debug:
        use_wandb = False
    print('use wandb:',use_wandb)
    config.save_path = os.path.join(config.save_path,config.experiment_name) 
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)


    if local_rank == 0 and use_wandb==True:
        wandb_writer = WandbWriter(exp_name=f"MA_{config.experiment_name}",
                                cfg=config,
                                output_dir='./logs/wandb'
                                ) 
    else:
        wandb_writer = None

    print(config)

    config_file = os.path.join(config.save_path, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(config), f, indent=4)
        print('save config to ',config_file)

    # set the seed value
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = config.dataset_path

    split = config.split
    print("dataset split",split)

    
    train_dataset = MultiAttrCompositionDataset(root=dataset_path,
                                                phase='train',
                                                split=split)


    # val_dataset = MultiAttrCompositionDataset(root=dataset_path,
    #                                             phase='val',
    #                                             split=split)

    test_dataset = MultiAttrCompositionDataset(root=dataset_path,
                                                phase='test',
                                                split=split)
    

    # if config.debug:
    #     train_dataset.data = train_dataset.data[0:200]          
    #     test_dataset.data = test_dataset.data[0:200]                   


    val_dataset = None
    # test
    # train_dataset.data = train_dataset.data[:10]
    # test_dataset.data = test_dataset.data[:10]


    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model,optimizer = get_model(train_dataset,config,device)

    # resume training
    config.load_training = False
    if config.load_model and config.load_training:
        print("resume training from ",config.load_model)
        print("Load model from ",config.load_model)
        model.load_state_dict(torch.load(config.load_model))


        optimizer_path = config.load_model.replace("epoch_", "optimizer_epoch_")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
            print(f"Loaded optimizer state from {optimizer_path}")
        else:
            print(f"Warning: optimizer state not found at {optimizer_path}")


    os.makedirs(config.save_path, exist_ok=True)

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset,wandb_writer)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")

