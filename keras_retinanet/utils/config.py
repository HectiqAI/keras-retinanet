"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import configparser
import numpy as np
import keras
from ..utils.anchors import AnchorParameters


def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)


import os
import datetime

class RetinaConfig():
    
    PRIOR_PROBABILITY = 0.01
    
    USE_MULTIPROCESSING = False
    MAX_QUEUE_SIZE = 8
    
    # Number of image for training
    IMAGE_LIMIT = 999999
    
    # Clip norm
    CLIP_NORM = 1e-1
    
    # Learning rate
    LR_SCHEDULE = [(0,1e-4)]
    
    # Work station
    WORK_STATION = "/home/edwardl/trainings/instrumments/retinanet/"
    
    def __init__(self):
        self.set_log_dir()
        return
    def save_config(self):
        return
    def load_config(self):
        return
    def checkpoint_path(self):
        return
    
    @property
    def tensorboard_dir(self):
        os.path.join(self.WORK_STATION, self.NAME)
        return 
    
    def set_log_dir(self):
        
        now = datetime.datetime.now()
        self.log_dir = os.path.join(self.WORK_STATION, "{}{:%Y%m%dT%H%M}".format(
            self.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.WORK_STATION, "_retinanet_{}_*epoch*.h5".format(
            self.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
    
    def _schedule(self, epoch):
        for item in self.LR_SCHEDULE:
            t = item[0]
            if epoch>=t:
                lr = item[1]
        return lr
                
    def learning_rate_scheduler(self):
        return keras.callbacks.LearningRateScheduler(self._schedule, verbose=0)

