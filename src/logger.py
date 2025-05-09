import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir='outputs/logs', experiment_name=None):
        if experiment_name is None:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"experiment_{time_str}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, img_tensor, step):
        self.writer.add_image(tag, img_tensor, step)
    
    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, step)
    
    def log_model_graph(self, model, input_tensor):
        self.writer.add_graph(model, input_tensor)
    
    def log_hyperparams(self, hparam_dict, metric_dict):
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        self.writer.close()