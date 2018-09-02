import os
import time
from inference import Inference
from hourglass import HourglassModel
from datagen import DataGenerator
from utils import process_config
import numpy as np
import tensorflow as tf
import configparser
import cv2
from predict_all import PredictAll

tf.app.flags.DEFINE_string("model_dir", "trained/hg_stack1_de", "pose model directory")
tf.app.flags.DEFINE_string("config_file", "config_dlib.cfg", "config file name")
tf.app.flags.DEFINE_string("model_file", "hourglass_100", "pose model file name")
tf.app.flags.DEFINE_boolean("resize", False, "whether to resize the image to 256*256 directly")
tf.app.flags.DEFINE_boolean("hm", False, "whether to show the heat maps")
tf.app.flags.DEFINE_string("image_file", None, "image file name")
tf.app.flags.DEFINE_integer("camera", None, "whether to predict camera")
tf.app.flags.DEFINE_string("video", None, "whether to predict video")
tf.app.flags.DEFINE_string("video_save", None, "whether to save video predict result")

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':
	print('--Parsing Config File')

	modeldir = FLAGS.model_dir
	configfile = os.path.join(modeldir, FLAGS.config_file)
	modelfile = os.path.join(modeldir, FLAGS.model_file)
	print(modelfile)

	params = process_config(configfile)
	model = Inference(params=params, model=modelfile)
	predict = PredictAll(model=model, resize=FLAGS.resize, hm=FLAGS.hm)

	if FLAGS.image_file is not None:
		# single image prediction
		predict.predict_image(params['img_directory'], FLAGS.image_file)
	elif FLAGS.camera is not None:
		predict.predict_camera(FLAGS.camera)
	elif FLAGS.video is not None:
		predict.predict_video(FLAGS.video, FLAGS.video_save)
