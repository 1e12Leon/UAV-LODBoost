import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
from BotSort_tracker.tracker.bot_sort import BoTSORT
from BotSort_tracker.visualize import plot_tracking


def botsort_demo(yolo, video_path, video_save_path, video_fps):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    tracker = BoTSORT()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # 将frame转换为Image读取的形式并检测
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = yolo.detect_image_dets(image)

            # 将dets转换为array形式，并用cv2读取图片，botsort要求输入图片为cv2的格式
            dets = np.array(dets)

            # 进行botsort
            online_targets = tracker.update(dets, frame)

            results = []
            online_tlwhs = []
            online_ids = []
            online_scores = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            online_im = plot_tracking(frame, online_tlwhs, online_ids)
            out.write(online_im)
            cv2.imshow("Frame", online_im)
        else:
            break
    cap.release()
    out.release()