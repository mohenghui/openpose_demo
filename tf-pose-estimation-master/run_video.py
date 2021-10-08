import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='./images/cxk.mp4')  #改为自己的输入视频
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--show-process', type=bool, default=False,help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    ret_val, image = cap.read()
    video_size = (image.shape[1],image.shape[0])
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    #video_size = (2560,1920)
    outVideo = cv2.VideoWriter('save.avi', fourcc, 10, video_size)
    filename = './images/cxk2.mp4'    #更改为自己的保存路径
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()

        if not ret_val:
            break
        logger.debug('image process+')
        #image = common.read_imgfile(args.image, None, None)
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        logger.debug('show+')
        # image_res = cv2.resize(image,(640,480))
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        outVideo.write(image)
        # video_write = cv2.VideoWriter(filename, fourcc, 10, video_size)
        # video_write.write(image)

        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    cap.release()
    outVideo.release()
    cv2.destroyAllWindows()
    '''   while cap.isOpened():
        ret_val, image = cap.read()

        humans = e.inference(image,resize_to_default=(w > 0 and h > 0))
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
'''
