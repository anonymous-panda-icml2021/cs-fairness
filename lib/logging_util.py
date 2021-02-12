import tempfile

import numpy as np
import torch
import tensorflow as tf

class Logger:
    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir)

    def log_scalar(self, tag, val, step, flush=True):
        if hasattr(val, 'item'):
            val = val.item()
        with self.writer.as_default():
            tf.summary.scalar(tag, val, step=step)
        if flush: self.writer.flush()

    def log_scalars(self, tag_value_dict, step):
        for tag, val in tag_value_dict.items():
            self.log_scalar(tag, val, step, flush=False)
        self.writer.flush()

    def log_image(self, tag, img, step, max_outputs=10, description=None, flush=True):
        if isinstance(img, torch.Tensor):
            # N, C, H, W -> N, H, W, C
            data = tf.constant(img.cpu().permute(0,2,3,1).numpy())
        elif isinstance(img, tf.Tensor):
            data =  img
        else:
            raise ValueError(f'Unsupported data type {type(img)}')

        with self.writer.as_default():
            tf.summary.image(tag, data, step=step,
                             max_outputs=max_outputs,
                             description=description)
        if flush: self.writer.flush()

    def log_images(self, tag_image_dict, step, max_outputs=10, description=None):
        for tag, img in tag_image_dict.items():
            self.log_image(tag, img, step, max_outputs=max_outputs, description=description)
        self.writer.flush()

def test_logger():
    logger = Logger('.')

if __name__ == '__main__':
    test_logger()
