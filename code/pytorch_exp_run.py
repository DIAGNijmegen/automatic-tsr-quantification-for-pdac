import torch_data_generator as data_generator

import time
import gc
import os 
import sys 
import shutil 
import numpy as np 
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp

import numpy as np
import yaml
from torch.utils.data import Dataset
from torchinfo import summary

import wandb 

def collect_arguments():
    """
    Collect command line arguments.
    """
    argument_parser = argparse.ArgumentParser(description='Torch train run.')
    argument_parser.add_argument('-p', '--project_name', required=False, type=str, default="model",help='project name in wandb')
    argument_parser.add_argument('-d', '--data_path', required=True, type=str, help='input data file')
    argument_parser.add_argument('-c', '--config_path', required=True, type=str, help='input')
    argument_parser.add_argument('-a', '--alb_config_path', required=False, type=str, default="", help='albumentations')
    argument_parser.add_argument('-o', '--output_path', required=True, type=str, help='output')
    arguments = vars(argument_parser.parse_args())

    return arguments["project_name"], arguments["data_path"], arguments["config_path"], \
           arguments["alb_config_path"], arguments["output_path"]
           
           
def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )

def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_model(model_param: dict, training_param:dict, num_classes: int):
    print("Init model...")

    if model_param['modelname'] == 'unet':
        model = smp.Unet(encoder_name=model_param['backbone'], 
                         classes=num_classes, 
                         encoder_weights=model_param['encoder_weights']
                         )

    elif model_param['modelname'] == 'unet-plus':
        model = smp.UnetPlusPlus(encoder_name=model_param['backbone'], 
                                classes=num_classes, 
                                encoder_weights=model_param['encoder_weights']
                                )

    if model_param['loss'] == 'cc':
        loss_fn = nn.NLLLoss(ignore_index=-100)
    elif model_param['loss'] == 'lovasz':
        loss_fn = smp.losses.LovaszLoss(mode='multiclass', ignore_index=-100)
    elif model_param['loss'] == 'dice':
        loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=-100)

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=model_param['learning_rate']),]) 

    if model_param['learning_rate_schedule'] == 'plateau':
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min',
                                                               factor=training_param['lr_reduction_factor'], 
                                                               patience=training_param['lr_plateau'])

    elif model_param['learning_rate_schedule'] == 'one_cycle':
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=0.001, 
                                                        steps_per_epoch=1, 
                                                        epochs=training_param['epochs'],
                                                        final_div_factor=1e6)                                            

    return model, loss_fn, optimizer, schedular


def get_metrics(predictions: torch.Tensor, 
                targets: torch.Tensor, 
                mode: str = 'multiclass', 
                threshold:float = 0.5, 
                num_classes=2) -> dict:
    """Create metrics to monitor network performance
    
    First compute statistics for true positives, false positives, false negative and true negative "pixels"
    Then calculate measures with required reduction (see metric docs)
    Args:
        predictions (torch.Tensor): _description_
        targets (torch.Tensor): _description_
        threshold (0.5float): _description_
        mode (_type_, optional): _description_. Defaults to 'multilabel':str.

    Returns:
        dict: _description_
    """

    #TODO: check if this is the best option to calculate the values 
    tp, fp, fn, tn = smp.metrics.get_stats(predictions.argmax(dim=1), 
                                           targets, 
                                           mode=mode, 
                                           ignore_index=-100, 
                                           num_classes=num_classes)

    
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").detach().cpu().numpy()
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").detach().cpu().numpy()
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro").detach().cpu().numpy()
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").detach().cpu().numpy()
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").detach().cpu().numpy()     
    
    return iou_score, f1_score, f2_score, accuracy, recall


def train_loop(data_gen, model, loss_fn, scaler, optimizer, device, use_amp, num_classes):
    
    epoch_loss, epoch_iou, epoch_f1, epoch_f2, epoch_acc, epoch_recall = list(), list(), list(), list(), list(), list()
    softmax_fn = nn.LogSoftmax(dim=1)   
    
    for tbatch_idx, (inputs, targets) in enumerate(data_gen): 

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions = model.forward(inputs)
                if isinstance(loss_fn, torch.nn.NLLLoss): 
                    loss = loss_fn(softmax_fn(predictions), targets.squeeze())
                else: 
                    loss = loss_fn(predictions, targets.squeeze())
                    
            optimizer.zero_grad()
            
            if scaler:     
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: 
                loss.backward()
                optimizer.step()
            
            iou_score, f1_score, f2_score, accuracy, recall = get_metrics(softmax_fn(predictions), 
                                                                          targets.squeeze(), 
                                                                          num_classes=num_classes)
            loss = loss.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()    
            
            epoch_loss.append(loss)
            epoch_iou.append(iou_score)
            epoch_f1.append(f1_score)
            epoch_f2.append(f2_score)
            epoch_acc.append(accuracy)
            epoch_recall.append(recall)
    
    garbage_collection_cuda()

    return {'training_loss': np.sum(epoch_loss) / (tbatch_idx +1), 
            'training_iou_score': np.sum(epoch_iou) / (tbatch_idx +1), 
            'training_f1_score': np.sum(epoch_f1) / (tbatch_idx +1), 
            'training_f2_score': np.sum(epoch_f2) / (tbatch_idx +1), 
            'training_accuracy': np.sum(epoch_acc) / (tbatch_idx +1), 
            'training_recall': np.sum(epoch_recall) / (tbatch_idx +1)}


def validation_loop(data_gen, model, loss_fn, device, use_amp, num_classes):

    epoch_loss, epoch_iou, epoch_f1, epoch_f2, epoch_acc, epoch_recall = list(), list(), list(), list(), list(), list()
    softmax_fn = nn.LogSoftmax(dim=1)  
    
    for tbatch_idx, (inputs, targets) in enumerate(data_gen): 

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    predictions = model.forward(inputs)
                    if isinstance(loss_fn, torch.nn.NLLLoss): 
                        loss = loss_fn(softmax_fn(predictions), targets.squeeze())
                    else: 
                        loss = loss_fn(predictions, targets.squeeze())
                
                iou_score, f1_score, f2_score, accuracy, recall = get_metrics(softmax_fn(predictions), 
                                                                              targets.squeeze(), 
                                                                              num_classes=num_classes)
                loss = loss.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()    
                
                epoch_loss.append(loss)
                epoch_iou.append(iou_score)
                epoch_f1.append(f1_score)
                epoch_f2.append(f2_score)
                epoch_acc.append(accuracy)
                epoch_recall.append(recall)
    
    garbage_collection_cuda()
    
    return {'validation_loss': np.sum(epoch_loss) / (tbatch_idx +1), 
            'validation_iou_score': np.sum(epoch_iou) / (tbatch_idx +1), 
            'validation_f1_score': np.sum(epoch_f1) / (tbatch_idx +1), 
            'validation_f2_score': np.sum(epoch_f2) / (tbatch_idx +1), 
            'validation_accuracy': np.sum(epoch_acc) / (tbatch_idx +1), 
            'validation_recall': np.sum(epoch_recall) / (tbatch_idx +1)}

def save_model(model, output_path: str):
    # torch.save({'epoch': epoch_indx, 
    #             'model_state_dict': model.state_dict(), 
    #             'optimizer_state_dict': optimizer.state_dict(), 
    #             'loss', loss}, output_path)
    
    torch.save(model.state_dict(), output_path)


def get_config_from_yaml(config_path: str) -> dict:
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)
    return parameters['model'], parameters['sampler'], parameters['training']


def main():
    project_name, data_path, config_path, albumentations_path, output_path = collect_arguments()

    model_parameters, sampler_parameters, training_parameters = get_config_from_yaml(config_path)    
    
    logger = wandb.init(project=project_name, entity="pierpaolov")
    shutil.copyfile(config_path, os.path.join(output_path, project_name + '_' + logger.name + '.yaml'))

    print("Init data loaders...")
    datawrapper = data_generator.PtDataLoader(data_path, 
                                              albumentations_path, 
                                              sampler_parameters, 
                                              training_parameters)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = training_parameters['mixed_precision']
    num_classes = len(np.unique(list(sampler_parameters['training']['label_map'].values())))
    
    model, loss_fn, optimizer, schedular = init_model(model_parameters, 
                                                      training_parameters, 
                                                      num_classes)

    wandb.config = {"learning_rate":model_parameters['learning_rate'], 
                    "backbone": model_parameters["backbone"],
                    "encoder_weights": model_parameters["encoder_weights"],
                    "epochs": training_parameters['epochs'],
                    "training_batch_size": training_parameters['training_batch_size']}
    wandb.watch(model)
    
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
    model.to(device)

    garbage_collection_cuda()
    
    print("Start training...")

    best_val_loss = np.inf

    save_paths_list = list() 

    for epoch_indx in range(training_parameters['epochs']):
        start_time = time.time()
        model.train()
        
        train_results = train_loop(datawrapper.training_generator, 
                                   model, 
                                   loss_fn, 
                                   scaler, 
                                   optimizer, 
                                   device, 
                                   use_amp,
                                   num_classes)

        model.eval()
        
        validation_results = validation_loop(datawrapper.validation_generator,
                                             model, 
                                             loss_fn, 
                                             device, 
                                             use_amp, 
                                             num_classes)
        
        if model_parameters['learning_rate_schedule'] == 'plateau':
            schedular.step(validation_results['validation_loss'])

        elif model_parameters['learning_rate_schedule'] == 'one_cycle':
            schedular.step()


        garbage_collection_cuda()
        update_dict = {**train_results, **validation_results}
        update_dict['learning_rate'] = optimizer.state_dict()["param_groups"][0]["lr"]        
        wandb.log(update_dict)

        # We save the last 10 epochs
        # 
        if len(save_paths_list) > 10: 
            os.remove(save_paths_list[0])
            save_paths_list.pop(0)

        save_paths_list.append(os.path.join(output_path, project_name + '_{}_epoch_{}.pt'.format(logger.name, epoch_indx + 1)))

        save_model(model, os.path.join(output_path, project_name + '_{}_epoch_{}.pt'.format(logger.name, epoch_indx + 1)))
        
        if update_dict['validation_loss'] < best_val_loss:
            best_val_loss = update_dict['validation_loss']
            save_model(model, os.path.join(output_path, project_name + '_{}_best_model.pt'.format(logger.name)))

        end_time = time.time() 
        total_time = (end_time - start_time) / 60 if (end_time - start_time) > 60 else (end_time - start_time)
        time_string = 'minutes' if (end_time - start_time) > 60 else 'seconds'
        print('Epoch {:d} train_loss {:.4f} val_loss {:.4f} total time {:.4f} {}'.format(epoch_indx+1, 
                                                                                         train_results['training_loss'], 
                                                                                         validation_results['validation_loss'],
                                                                                         total_time, 
                                                                                         time_string))
        
        datawrapper.on_epoch_end()
    wandb.finish()


if __name__ == '__main__':
    sys.exit(main())