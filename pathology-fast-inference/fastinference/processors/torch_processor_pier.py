import os
from importlib import import_module

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml

from ..async_tile_processor import async_tile_processor


class torch_processor_pier(async_tile_processor): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.softmax_fn = nn.LogSoftmax(dim=1) 

    def get_config_from_yaml(self, config_path: str) -> dict:
        with open(file=config_path, mode='r') as param_file:
            parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)
        return parameters['model'], parameters['sampler'], parameters['training']          

    def _load_network_model(self):
        models = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # config_path = self._model_path.split("_epoch")[0].split("_best")[0] + '.yaml'
        # print(f'Model path/directory: {self._model_path}')
        if os.path.isdir(self._model_path):
            model_files = sorted([file for file in os.listdir(self._model_path) if file.endswith('.pt')])
            
            for model_file in model_files:
                model_file_path = os.path.join(self._model_path, model_file)

                print(f'Model_file_path:{model_file}')
                config_path_name = model_file.replace('_best_model.pt','.yaml')
                config_path = os.path.join(self._model_path,config_path_name)
                print(f'Config_path: {config_path_name}')

                model_parameters, sampler_parameters, training_parameters = self.get_config_from_yaml(config_path)
                n_classes = len(np.unique(list(sampler_parameters['training']['label_map'].values())))
    
                model = smp.Unet(encoder_name=model_parameters['backbone'], 
                        classes=n_classes, 
                        encoder_weights=model_parameters['encoder_weights'])

                state_dict = torch.load(model_file_path)
                model.load_state_dict(state_dict)
                print("Model succesfully loaded!")
                model.to(device)
                model.eval()
                models.append(model) 


        return models           

    def _predict_tile_batch(self, tile_batch=None, info=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        
        tile_batch = torch.from_numpy(tile_batch).to(device)
        preds = []
        with torch.inference_mode():  # Disable gradient tracking for inference
            with torch.cuda.amp.autocast():   
                for model in self._model:   
                    preds.append(model.predict(tile_batch))

        result = torch.mean(torch.stack(preds), dim=0)

            


            # result = self._model.predict(tile_batch)
            
        result = self.softmax_fn(result)  
        result = result.detach().cpu().numpy()

        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        
        return result

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info',
                                   '',1))
