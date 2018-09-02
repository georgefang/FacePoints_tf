"""
Face Alignment

"""

import configparser
from hourglass import HourglassModel
from datagen import DataGenerator
from utils import process_config
import tensorflow as tf
import os

tf.app.flags.DEFINE_string("configfile", "config/config_dlib.cfg", "config file name")
tf.app.flags.DEFINE_string("loadmodel", None, "whether to use center scale method")

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config( FLAGS.configfile )
	os.system('mkdir -p {}'.format(params['saver_directory']))
	os.system('cp {0} {1}'.format(FLAGS.configfile, params['saver_directory']))
	
	print('--Creating Dataset')
	dataset = DataGenerator(params['num_joints'], params['img_directory'], params['training_txt_file'], params['img_size'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel(params=params, dataset=dataset, training=True)
	model.generate_model()
	# model.restore('trained/tiny_200/hourglass_tiny_200_200')
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset=None, load='example/models/stack1_mobile_f32_de/hourglass_normal_100')
	
