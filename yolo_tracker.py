import os
import numpy as np
from PIL import Image
import cv2
import time
from tracker.byte_tracker import BYTETracker
from utils.visualize import plot_tracking
from tracking_utils.timer import Timer

from yolo import YOLO


def track_demo(yolo, video_path, video_save_path, video_fps, txt_dir):
    # ---------------------------------------------------------------------#
    #   tracker video setting
    # ---------------------------------------------------------------------#
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    # print(deteted)
    # yolo = YOLO()
    # Tracking
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.7
    frame_rate = 25
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20_check = False
    res_file = os.path.join(txt_dir, video_path.split('/')[-1].replace('mp4', 'txt'))

    print(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)
    timer = Timer()
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    frame_id = 0
    results = []
    while True:
        _, im0 = cap.read()
        frame_id += 1
        if _:
            height, width, _ = im0.shape
            t1 = time.time()
            frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            dets = yolo.detect_image_dets(frame)
            # dets = deteted.detecte(im0)
            # ---------------------------------------------------------------------#
            #
            # ---------------------------------------------------------------------#
            if np.array(dets).shape != (0,):
                online_targets = tracker.update(np.array(dets), [height, width], (height, width))
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save result for evaluation
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                t2 = time.time()
                print(f"FPS:{1 / (t2 - t1):.2f}")
                timer.toc()
                # print(1. / timer.average_time)
                online_im = plot_tracking(im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / 1 / (t2 - t1))
                with open(res_file, 'w') as f:
                    f.writelines(results)
                if video_save_path != "":
                    out.write(online_im)
                cv2.imshow("Frame", online_im)

            else:
                if video_save_path != "":
                    out.write(im0)
                    print("no target!")
                cv2.imshow("Frame", im0)

            ch = cv2.waitKey(1)  # 1 millisecond
            if ch == ord("q"):
                break

            if ch == 27:
                cap.release()
                break
        else:
            break


if __name__ == "__main__":
    yolo = YOLO()
    video_path = "video/test_person.mp4"  # 跟踪视频
    video_save_path = "video/out.mp4"  # 跟踪结果视频保存路径
    video_fps = 30.0
    txt_dir = "result"  # 跟踪结果报错路径
    track_demo(yolo, video_path, video_save_path, video_fps, txt_dir)
