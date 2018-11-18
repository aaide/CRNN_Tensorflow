import argparse
import io
import os
from base64 import b64decode
from time import time
import flask
import logging
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

from crnn_model import crnn_model
from local_utils import log_utils, data_utils
from local_utils.config_utils import load_config


logger = log_utils.init_logger()
logging.basicConfig(level=logging.INFO)

app = flask.Flask(__name__, instance_relative_config=True)
app.config.from_mapping(MODEL_VERSION="2018-11-18")


@app.route('/detect', methods=['POST'])
def detect():
    """
    """
    data = flask.request.get_json()
    logging.info("Request received for %d images" % len(data.get('images', {})))

    encoding = data.get('encoding', 'utf-8')
    images = {field: Image.open(io.BytesIO(b64decode(img_data))).convert("RGB")
              for field, img_data in data.get('images', {}).items()}
    
    result = {'model_version': app.config['MODEL_VERSION'],
              'encoding': encoding, 'contents': {}}

    t = time()
    for field, image in images.items():
        image = cv2.resize(np.array(image), tuple(cfg.ARCH.INPUT_SIZE))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        preds = sess.run(decodes, feed_dict={inputdata: image})

        preds = codec.writer.sparse_tensor_to_str(preds[0])

        logger.info('Field {:s}, detected label {:s}'.format(field, preds[0]))

        if np.any(preds is None):
            logging.info("Failed detection for %s" % field)
            continue
        result['detections'][field] = str(preds[0])

    result['detections_time'] = time() - t
    return flask.jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use',
                        default='model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999')
    parser.add_argument('-f', '--config_file', type=str,
                        help='Use this global configuration file')
    parser.add_argument('-c', '--chardict_dir', type=str,
                        help='Directory where character dictionaries for the '
                             'dataset were stored')
    parser.add_argument('-n', '--num_classes', type=int, default=0,
                        help='Force number of character classes to this number.'
                             ' Set to 0 for auto (read from charset_dir)')
    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.chardict_dir:
        config.cfg.PATH.CHAR_DICT_DIR = args.chardict_dir

    cfg = config.cfg

    t = time()
    logging.info("Starting service")

    w, h = cfg.ARCH.INPUT_SIZE
    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, h, w, cfg.ARCH.INPUT_CHANNELS],
                               name='input')

    codec = data_utils.TextFeatureIO(
        char_dict_path=os.path.join(cfg.PATH.CHAR_DICT_DIR, 'char_dict.json'),
        ord_map_dict_path=os.path.join(cfg.PATH.CHAR_DICT_DIR, 'ord_map.json'))

    num_classes = len(codec.reader.char_dict) + 1 if args.num_classes == 0 \
        else args.num_classes

    net = crnn_model.ShadowNet(phase='Test',
                               hidden_nums=cfg.ARCH.HIDDEN_UNITS,
                               layers_nums=cfg.ARCH.HIDDEN_LAYERS,
                               num_classes=num_classes)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)

    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out,
                                               sequence_length=cfg.ARCH.SEQ_LENGTH * np.ones(1),
                                               merge_repeated=False)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)
    with sess.as_default():
        saver.restore(sess=sess, save_path=args.weights_path)
        logging.info("Model loaded in %.2d seconds" % (time() - t))
        app.run(port=args.port, host=args.host)

    sess.close()
