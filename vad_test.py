# coding:utf8
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import argparse
import prettytable as pt

parser = argparse.ArgumentParser()
parser.add_argument("model_conf", help="json file that config model structure", type=str)
parser.add_argument("test_dir", help="contains wav.scp, i.e., the list of audio files to be tested", type=str)
parser.add_argument("model_dir", help="dir that stores the model", type=str)
parser.add_argument("--feat_type", help="feature type", type=str, default="fbank", choices=['fbank', 'mfcc'])
parser.add_argument("--front_context", help="context frame number", type=int, default=5)
parser.add_argument("--end_context", help="context frame number", type=int, default=5)
parser.add_argument("--start_threshold", help="speech start threshold", type=float, default=0.7)
parser.add_argument("--end_threshold", help="speech end threshold", type=float, default=0.6)
parser.add_argument("--lopass", help="low pass for feature extraction", type=int, default=100)
parser.add_argument("--hipass", help="the hipass for fbank feature", type=int, default=8000)
parser.add_argument("--epoch", help="test on which epoch", type=int, default=0)
parser.add_argument("--skip", help="skip frame or not", type=int, default=1)
parser.add_argument("--process_num", help="number of process that run parallel", type=int, default=20)
parser.add_argument("--gpu_index", help="which gpu to use", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument("--result_dir", help="dir to store vad test result", type=str, default="./vad_result")
parser.add_argument("--label_dir", help="dir to store vad test label", type=str, default="./vad_label")


args = parser.parse_args()

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from tools import *

init_log(args.test_dir)

# >>>>>>>>>>>>>>>>> print argments
for arg in vars(args):
    INFO('{} = {}'.format(arg, getattr(args, arg)), print_option=False)

# >>>>>>>>>>>>>>>> gpu device config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
import tensorflow as tf

INFO("Tensorflow version: {} (eager={})".format(tf.__version__, tf.executing_eagerly()), print_option=True)
if not tf.test.is_gpu_available():
    ERROR("No available GPUs")

strategy = tf.distribute.MirroredStrategy()

# >>>>>>>>>>>>>>>>> prepare and check inputs
test_dir = os.path.abspath(args.test_dir)
model_dir = os.path.abspath(args.model_dir)
wav_scp_fpath = os.path.join(test_dir, 'wav.scp')
cmvn_fpath = os.path.join(model_dir, 'final', 'cmvn')

# >>>>>>>>>>>>>>>>>>> prepare feature extractor
feat_extractor = FeatureWrapper(args.lopass, args.hipass)
feat_type = args.feat_type

FRONT_CONTEXT = args.front_context
END_CONTEXT = args.end_context
FEATURE_DIM = feat_extractor.fbank_dim
WINDOW_SIZE = FRONT_CONTEXT + END_CONTEXT + 1
OUT_SIZE = 0

if (not os.path.exists(cmvn_fpath)) \
        or (not os.path.exists(wav_scp_fpath)):
    ERROR(f"{wav_scp_fpath} or {cmvn_fpath} does not exist")


# >>>>>>>>>>>>>>>>  prepare dnn input
def load_cmvn(cmvn_scp_fpath):
    with open(cmvn_scp_fpath, 'r') as fin:
        mean_str = fin.readline()
        var_str = fin.readline()
    ''' 
    cmvn mat: a numpy array containing the mean and variance statistics. The
        first row contains the sum of all the fautures and as a last element
        the total number of features. The second row contains the squared
        sum of the features and a zero at the end
    '''
    cmvn_mean = np.asarray([float(x) for x in mean_str.split(' ')[:-1]], dtype=np.float32)
    cmvn_std = np.sqrt(np.asarray([float(x) for x in var_str.split(' ')[:-1]], dtype=np.float32) - np.square(cmvn_mean))
    return cmvn_mean, cmvn_std

def make_dnn_input(utt_mat, cmvn_mean, cmvn_std):
    if cmvn_mean.shape[0] != utt_mat.shape[1] or \
        cmvn_std.shape[0] != utt_mat.shape[1]:
        ERROR("[mor_test] make_dnn_input(), input cmvn shape not match")
    # apply cmvn
    feats = np.divide(np.subtract(utt_mat, cmvn_mean), cmvn_std)
    # concat features in the context window
    paddings = ((FRONT_CONTEXT, END_CONTEXT), (0, 0))
    feats = np.pad(feats, paddings, "constant")
    feats_list = [feats[i:i + utt_mat.shape[0], :] for i in range(WINDOW_SIZE)]
    feats = np.concatenate(feats_list, axis=1)
    return feats

cmvn_mean, cmvn_std = load_cmvn(cmvn_fpath)

# >>>>>>>>>>>>>>>> load model
with strategy.scope():
    with open(args.model_conf, "r") as json_file:
        model_conf = json.load(json_file)
    model_handler = ModelHandler(model_conf)
    model = model_handler.model
    OUT_SIZE = model_handler.output_shape()[1]
    if WINDOW_SIZE != model_handler.context_window():
        FATAL("feature window size {} != model window size {}".format(WINDOW_SIZE, model_handler.context_window()))
    print("window size: {}, out size: {}".format(WINDOW_SIZE, OUT_SIZE))

    if args.epoch == 0:
        latest = tf.train.latest_checkpoint(model_dir)
    elif args.epoch > 0:
        latest = os.path.join(model_dir, "ckpt_{}".format(args.epoch))
        if not os.path.exists(latest + ".index"):
            ERROR("input checkpoint file not exist")
    model.load_weights(latest)


# >>>>>>>>>>>>>>>> vad decode
from enum import Enum
from collections import deque

class vad_state(Enum):
    SIL     = 0
    START   = 1
    SPEECH  = 2
    PAUSE   = 3
    END     = 4

class vad_decoder():

    def __init__(self, conf):
        self._state = vad_state.SIL
        self._scores = deque()
        self._start_threshold = 1- conf["vad_start_threshold"]
        self._end_threshold = 1 - conf["vad_end_threshold"]
        self._pause_frame = conf["vad_pause_max_frame"]
        self._valid_frame = conf["vad_valid_min_frame"]
        self._smooth_frame = conf["vad_smooth_frame"]
        self._duration = 0

    def reset(self):
        self._state = vad_state.SIL
        self._duration = 0

    def decode_once(self, current_frame_sil_score):
        self._scores.append(current_frame_sil_score)
        if len(self._scores) > self._smooth_frame:
            self._scores.popleft()
        current_smoothed_sil_score = sum(self._scores) / len(self._scores)
        if vad_state.SIL == self._state:
            if current_smoothed_sil_score < self._start_threshold:
                self._state = vad_state.START
        elif vad_state.START == self._state:
            self._state = vad_state.SPEECH
        elif vad_state.SPEECH == self._state:
            if current_smoothed_sil_score > self._end_threshold:
                self._state = vad_state.PAUSE
                self._duration = 0
        elif vad_state.PAUSE == self._state:
            self._duration += 1
            if current_smoothed_sil_score > self._end_threshold and self._duration > self._pause_frame:
                self._state = vad_state.END
            elif current_smoothed_sil_score <= self._end_threshold:
                self._state = vad_state.SPEECH
        elif vad_state.END == self._state:
            self.reset()

    def decode(self, scores):
        last_start_frame = 0
        result = []
        for frame, score in enumerate(scores):
            # print("frame {} sil score = {}".format(frame, score[0]))
            self.decode_once(score[0])
            if vad_state.START == self._state:
                last_start_frame = frame
            elif vad_state.END == self._state and frame - last_start_frame > self._pause_frame + self._valid_frame:
                result.append([last_start_frame / 100, frame / 100])
        return result

# >>>>>>>>>>>>>>>> test model
vad_conf = {"vad_start_threshold": args.start_threshold,
            "vad_end_threshold": args.end_threshold,
            "vad_pause_max_frame": 130,
            "vad_valid_min_frame": 15,
            "vad_smooth_frame": 10}
print(args.end_threshold)
decoder = vad_decoder(vad_conf)

def get_statistics(utt_id, result):
    """
    :param utt_id: looks like xxx_senario_id (senario == ori 即为label)
    :param result: [[start, end], ...]
    :return:
    """
    try:
        tmp = utt_id.split('_')
        # tmp[-2] = 'ori'
        label_file_name = '_'.join(tmp) + '.csv'
        with open(os.path.join(args.label_dir, label_file_name), "r") as label_file:
            _labels = label_file.readlines()
    except IOError as e:
        print(e)
        return None

    del _labels[0]
    labels = []
    for label in _labels:
        tmp = label.strip().split('\t')
        # print("tmp",tmp)
        if ':' in tmp[1]:
            start_list = tmp[1].split(':')
            start = float(start_list[0]) * 60 + float(start_list[1])
        else:
            start = float(tmp[1])

        if ':' in tmp[2]:
            end_list = tmp[2].split(':')
            end = start + (float(end_list[0]) * 60 + float(end_list[1]))
        else:
            end = start + float(tmp[2])

        # print("start:",start,"end",end)
        
        labels.append([start, end])
        

    label_id = 0
    max_err  = 1.2
    total_label_num = len(labels)
    correct_num = 0
    miss_num = 0
    # print("labels",labels)
    # print("result", result)
    for sample in result:
        while label_id < total_label_num and labels[label_id][0] < sample[0] + max_err:
            
            # if max(abs(labels[label_id][0] - sample[0]), abs(labels[label_id][1] - sample[1])) > max_err:
            if abs(labels[label_id][0] - sample[0]) > max_err:
                INFO("miss: {} {}".format(labels[label_id][0], labels[label_id][1]))
                miss_num += 1
            label_id += 1
        # print("label_id",label_id)
        label_id = max(label_id, 1)
        
        this_label = labels[label_id - 1]
        # if max(abs(this_label[0] - sample[0]), abs(this_label[1] - sample[1])) <= max_err:
        if abs(this_label[0] - sample[0]) <= max_err:
            correct_num += 1
            label_id += 1
        else:
            INFO("error: {} {}".format(sample[0], sample[1]))
        if label_id == total_label_num + 1:
            break
    
    
    if label_id < total_label_num + 1:
        miss_num += total_label_num + 1 - label_id
        while label_id < total_label_num + 1:
            INFO("miss: {} {}".format(labels[label_id - 1][0], labels[label_id - 1][1]))
            label_id += 1
    # print(total_label_num, len(result), correct_num, miss_num)
    return (total_label_num, len(result), correct_num, miss_num)


def draw_table(uttid, statistics):
    download_list = []
    
    if statistics is None:
        return
    label_num, predict_num, correct_num, miss_num = statistics
    # print("statistics",statistics)
    table = pt.PrettyTable()
    header = [uttid, 'True', 'False', 'Recall/Precision']

    table.field_names = header
    # print("label_num",label_num)
    label_num = max(label_num, 1)
    predict_num = max(predict_num, 1)
    table.add_row(['Positive', str(correct_num), str(predict_num - correct_num), 'Recall:'+str(int(10000*correct_num/label_num)/100)+'%'])
    table.add_row(['Negative', str(miss_num), '0', 'Precision:'+str(int(10000*correct_num/predict_num)/100)+'%'])
    
    recall_str = str(int(10000*correct_num/label_num)/100)+'%'
    precision_str = str(int(10000*correct_num/predict_num)/100)+'%'
    
    if uttid != "total":
        if recall_str != "100.0%" or precision_str != "100.0%":
            print("download:%s"%(uttid))
    
    # fw_out = open("/home/zhangzhicheng/Present_Code/Mor-Voice-Train-TF/test/download.list", 'w')
    # for file_name in download_list:
    #     fw_out.write("{}\n".format(file_name))

    INFO(table, print_option=True)

def run_decoder(utt_id, decoder_inputs, skip_flag):
    INFO("decoding audio file: {}".format(utt_id), print_option=True)
    # if skip frame, then assign scores of odd frame to the next even one
    if skip_flag != 0:
        odd_decoder_inputs = decoder_inputs[::2, :]
        concat_decoder_inputs = np.concatenate([odd_decoder_inputs, odd_decoder_inputs], axis=1)
        decoder_inputs = np.reshape(concat_decoder_inputs, (-1, OUT_SIZE))

    vad_result = decoder.decode(decoder_inputs)
    front_padding = 0.5
    for ii in vad_result:
        ii[0] = ii[0] - front_padding
    
    print("vad_result",vad_result)
    try:
        with open(os.path.join(args.result_dir, "{}.csv".format(utt_id)), "w") as result_csv:
            result_csv.write("Name\tStart\tDuration\tTime Format\tType\n")
            for i, seg in enumerate(vad_result):
                result_csv.write("{}\t{}\t{:.3f}\tdecimal\tCue\t\n".format(i, seg[0], seg[1] - seg[0]))
    except IOError:
        print("can't open {}".format(os.path.join(args.result_dir, "{}.csv".format(utt_id))))
    statistics = get_statistics(utt_id, vad_result)
    # draw_table(utt_id, statistics)
    # print("return statistics", statistics)
    return statistics


executor = ProcessPoolExecutor(max_workers=args.process_num)
with strategy.scope():
    all_task = []
    utt_list = []
    wakeup_count = {}

    with open(wav_scp_fpath, 'r') as fin:
        for line in fin:
            llist = line.strip().split()
            utt_id = llist[0]
            utt_list.append(utt_id)

            wav_fpath = llist[1]
            wav_data = feat_extractor.read_wav_16bit(wav_fpath)
            if feat_type == "fbank":
                feat_data = feat_extractor.get_fbank(wav_data)
            elif feat_type == "mfcc":
                feat_data = feat_extractor.get_mfcc(wav_data)
            model_inputs = make_dnn_input(feat_data, cmvn_mean, cmvn_std)
            decoder_inputs = model.predict(model_inputs)
            all_task.append(executor.submit(run_decoder, utt_id, decoder_inputs, args.skip))

    total_label_num, total_predict_num, total_corrent_num, total_miss_num = 0, 0, 0, 0
    for future in as_completed(all_task):
        data = future.result()
        if data is not None:
            label_num, predict_num, corrent_num, miss_num = data
            total_label_num += label_num
            total_predict_num += predict_num
            total_corrent_num += corrent_num
            total_miss_num += miss_num
    # draw_table("total", [total_label_num, total_predict_num, total_corrent_num, total_miss_num])
