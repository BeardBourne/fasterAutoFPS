from grabscreen import grab_screen
import win32api
import win32con
from operator import add

import keys as k
keys = k.Keys()

import pydirectinput

from datetime import datetime




import os
import sys
import json
import cv2
import math
import numpy as np
import paddle
from datetime import datetime

sys.path.insert(0, './PaddleDetection/deploy/python')
sys.path.insert(0, './PaddleDetection')
from deploy.python.det_keypoint_unite_utils import argsparser
from deploy.python.preprocess import decode_image
from deploy.python.infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images
from deploy.python.keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from deploy.python.visualize import draw_pose
from deploy.python.utils import get_current_memory_mb
from deploy.python.keypoint_postprocess import translate_to_ori_images



KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, det_threshold,
                           keypoint_threshold, run_benchmark):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res, det_threshold)
    keypoint_vector = []
    score_vector = []
    rect_vector = det_rects
    batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)

    for i in range(batch_loop_cnt):
        start_index = i * keypoint_batch_size
        end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
        batch_images = rec_images[start_index:end_index]
        batch_records = np.array(records[start_index:end_index])
        if run_benchmark:
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, warmup=10, repeats=10)
        else:
            keypoint_result = keypoint_detector.predict(batch_images,
                                                        keypoint_threshold)
        orgkeypoints, scores = translate_to_ori_images(keypoint_result,
                                                       batch_records)
        keypoint_vector.append(orgkeypoints)
        score_vector.append(scores)

    keypoint_res = {}
    keypoint_res['keypoint'] = [
        np.vstack(keypoint_vector).tolist(), np.vstack(score_vector).tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          keypoint_batch_size=1):

    monitor_width = 2560
    monitor_height = 1440

    screensz=800
    screen_region = (int((monitor_width-screensz)/2), int((monitor_height-screensz)/2), int((monitor_width+screensz)/2), int((monitor_height+screensz)/2))

    center = [int(monitor_width/2), int(monitor_height/2)]


    while True:
        if win32api.GetAsyncKeyState(win32con.VK_DIVIDE):
            print('cv2.destroyAllWindows()')
            cv2.destroyAllWindows()
            break

        if (win32api.GetAsyncKeyState(win32con.VK_MULTIPLY)):
            print('Loop start')
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

            screen = grab_screen(region=screen_region)
            window = screen

            image, _ = decode_image(window, {})

            print('detector')
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            results = detector.predict([image], FLAGS.det_threshold)
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

            if results['boxes_num'] == 0:
                continue
            
            # only target one person box
            results['boxes'] = np.array([results['boxes'][0]])
            results['boxes_num'] = np.array([1])

            print('topdown_keypoint_detector')
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            keypoint_res = predict_with_given_det(
                image, results, topdown_keypoint_detector, keypoint_batch_size,
                FLAGS.det_threshold, FLAGS.keypoint_threshold, FLAGS.run_benchmark)
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            # print(keypoint_res)
            if len(keypoint_res['bbox']) == 0:
                continue
            sum_x = 0
            sum_y = 0
            for i in range(0,5):
                sum_x += keypoint_res['keypoint'][0][0][i][0]
                sum_y += keypoint_res['keypoint'][0][0][i][1]


            target_center = [
                sum_x/5,
                sum_y/5
            ]
            target_center =list(map(add, [screen_region[0], screen_region[1]], target_center))
            # print(target_center)
            print('before pydirectinput.moveTo')
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

            # pydirectinput.moveTo(int(target_center[0]), int(target_center[1]))
            pydirectinput.click(int(target_center[0]), int(target_center[1]), clicks=0)
            # pydirectinput.leftClick(int(target_center[0]), int(target_center[1]))

            # pydirectinput.moveTo(int(target_center[0]), int(target_center[1]), relative=True)

            print('Loop end')
            print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

            # returnimg = draw_pose(
            #     screen,
            #     keypoint_res,
            #     visual_thread=FLAGS.keypoint_threshold,
            #     returnimg=True)
            # print('get returnimg!!')
            # cv2.imshow('result', returnimg)
            # print('after cv2.imshow!!')
            # cv2.waitKey(1)


def main():
    pred_config = PredictConfig(FLAGS.det_model_dir)
    detector_func = 'Detector'
    if pred_config.arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'

    detector = eval(detector_func)(pred_config,
                                   FLAGS.det_model_dir,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig_KeyPoint(FLAGS.keypoint_model_dir)
    assert KEYPOINT_SUPPORT_MODELS[
        pred_config.
        arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'
    topdown_keypoint_detector = KeyPoint_Detector(
        pred_config,
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.keypoint_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)


    # predict from image
    topdown_unite_predict(detector, topdown_keypoint_detector,
                            FLAGS.keypoint_batch_size)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
