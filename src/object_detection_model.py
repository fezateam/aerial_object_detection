import numpy as np
from importlib import import_module
from anomalib.config import get_configurable_parameters
from ensemble_boxes import weighted_boxes_fusion
from mmdet.apis import init_detector, inference_detector
from utils.plots import Annotator, colors
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2n, n2xyxy)
from utils.datasets import LoadImages
from models.experimental import attempt_load
import logging

import requests

from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject

import os
import sys
from pathlib import Path
import time
import cv2
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        yfile = open("config/config.yaml", "r").read()
        self.yfile = yaml.safe_load(yfile)

        tph_weights = "weights/yolov5l-xs-2.pt"
        imgsz = self.yfile["imgsz"]
        self.conf_thres = self.yfile["conf_thres"]
        self.iou_thres = self.yfile["iou_thres"]
        self.max_det = self.yfile["max_det"]
        self.names = self.yfile["names"]
        self.classes_num = self.yfile["classes_num"]
        self.save_img = self.yfile["save_img"]
        self.save_dir = self.yfile["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # n_config_file = self.yfile["n_config_file"]
        # n_checkpoint_file = self.yfile["n_checkpoint_file"]

        p_config_file = self.yfile["p_config_file"]
        p_checkpoint_file = self.yfile["p_checkpoint_file"]

        padim_config = get_configurable_parameters(
            config_path=self.yfile["padim_config_path"])
        padim_module = import_module("anomalib.deploy.inferencers.torch")
        TorchInferencer = getattr(padim_module, "TorchInferencer")
        self.uai_inferencer = TorchInferencer(
            config=padim_config, model_source=self.yfile["uai_inferencer_path"])
        self.uap_inferencer = TorchInferencer(
            config=padim_config, model_source=self.yfile["uap_inferencer_path"])

        self.device = torch.device(self.yfile["device"])
        self.stride, names = 64, [
            f'class{i}' for i in range(1000)]  # assign defaults
        # self.n_model = init_detector(
        #     n_config_file, n_checkpoint_file, device=self.yfile["device"])
        self.p_model = init_detector(
            p_config_file, p_checkpoint_file, device=self.yfile["device"])
        self.model_tph = attempt_load(tph_weights, map_location=self.device)
        self.stride = int(self.model_tph.stride.max())  # model stride
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.model_tph(torch.zeros(
            1, 3, *self.imgsz).to(self.device).type_as(next(self.model_tph.parameters())))  # run once

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(
            f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
        return image_name

    def process(self, prediction, evaluation_server_url):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        image = self.download_image(
            evaluation_server_url + "media" + prediction.image_url, "./_images/")
        dataset = LoadImages(
            f"./_images/{image}", img_size=self.imgsz, stride=self.stride, auto=".pt")

        for path, img, im0s, vid_cap, s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            pred_tph = self.model_tph(img, augment=False, visualize=False)[0]
            pred_tph = non_max_suppression(
                pred_tph, self.conf_thres, self.iou_thres, self.classes_num, False, max_det=self.max_det)

            for i, det in enumerate(pred_tph):
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                save_path = f"{self.save_dir}/{str(p.name)}"  # img.jpg
                height, width, channel = im0s.shape
                annotator = Annotator(
                    im0, line_width=1, example=str(self.names))
                # mm-object segment
                #n_result = inference_detector(self.n_model, im0s)
                mmo_threshold = self.yfile["mmo_threshold"]
                mmo_boxes = []
                mmo_confs = []
                mmo_labels = []
                # for e, mmo in enumerate(n_result):
                #     if len(mmo) > 0:
                #         if e != 0:
                #             continue
                #         for modet in mmo:
                #             if modet[-1] >= mmo_threshold:
                #                 box = list(
                #                     map(lambda x: round(float(x), 2) if x > 0 else 0, modet[:4]))
                #                 mmo_boxes.append(
                #                     xyxy2n([round(x, 3) for x in modet[:4]], width, height))
                #                 mmo_confs.append(round(modet[-1], 3))
                #                 mmo_labels.append(e)

                tpho_boxes = []
                tpho_confs = []
                tpho_labels = []
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    for tpho in det:
                        if tpho[-1]==2:
                            continue
                        
                        tpho = tpho.cpu().numpy()
                        box = list(map(lambda x: round(float(x), 2)
                                   if x > 0 else 0, tpho[:4]))
                        tpho_boxes.append(xyxy2n(box, width, height))
                        tpho_confs.append(round(float(tpho[4]), 2))
                        tpho_labels.append(int(tpho[-1]))

                boxes_list = [mmo_boxes, tpho_boxes]
                scores_list = [mmo_confs, tpho_confs]
                labels_list = [mmo_labels, tpho_labels]
                weights_list = self.yfile["weights_list"]
                boxes_n, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                                weights=weights_list, iou_thr=self.yfile["iou_thr_wbf"], skip_box_thr=self.yfile["skip_box_thr"])

                boxes = []
                scores = scores.tolist()
                labels = labels.astype(np.int8).tolist()
                if len(boxes_n):
                    for e, mbox in enumerate(boxes_n):
                        mbox = n2xyxy(mbox, width, height)
                        boxes.append(mbox)

                pist_score_list = [
                    "Inis Alani Degil" for x in range(len(boxes))]
                p_result = inference_detector(self.p_model, im0s)
                mmp_threshold = self.yfile["mmp_threshold"]
                for e, mmp in enumerate(p_result):
                    if len(mmp) > 0:
                        for mpdet in mmp:
                            if mpdet[-1] >= mmp_threshold:

                                box = list(map(lambda x: round(x, 2)
                                           if x > 0 else 0, mpdet[:4]))
                                boxes.append(box)
                                scores.append(round(mpdet[-1], 3))
                                labels.append(e+10)

                                padim_img = im0s[int(box[1]):int(
                                    box[3]), int(box[0]):int(box[2]), :]
                                padim_img = cv2.resize(padim_img, (224, 224))

                                if e == 0:
                                    padim_output = self.uap_inferencer.predict(
                                        image=padim_img, superimpose=False, overlay_mask=False)
                                    if isinstance(padim_output, tuple):
                                        anomaly_map, padim_score = padim_output

                                elif e == 1:
                                    padim_output = self.uai_inferencer.predict(
                                        image=padim_img, superimpose=False, overlay_mask=False)
                                    if isinstance(padim_output, tuple):
                                        anomaly_map, padim_score = padim_output

                                if padim_score >= self.yfile["padim_score"]:
                                    pist_score_list.append("Inilemez")
                                elif padim_score < self.yfile["padim_score"]:
                                    pist_score_list.append("Inilebilir")

                if len(boxes) > 0:
                    for xyxy, conf, cls in zip(boxes, scores, labels):
                        if self.save_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = (f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(
                                xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                # print(boxes)
            # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)

        frame_results = self.detect(prediction, boxes, labels, pist_score_list)
        return frame_results

    def detect(self, prediction, boxes, labels, pist_score_list):
        insan = self.yfile["insan"]
        tasit = self.yfile["tasit"]
        uap = self.yfile["uap"]
        uai = self.yfile["uai"]

        for e, box in enumerate(boxes):
            if labels[e] in insan:
                cls = classes["Insan"]
            elif labels[e] in tasit:
                cls = classes["Tasit"]
            elif labels[e] in uap:
                cls = classes["UAP"]
            elif labels[e] in uai:
                cls = classes["UAI"]
            else:
                cls = 1

            landing_status = landing_statuses[pist_score_list[e]]

            top_left_x = round(float(box[0]), 2) if box[0] > 0 else 0.
            top_left_y = round(float(box[1]), 2) if box[1] > 0 else 0.
            bottom_right_x = round(float(box[2]), 2) if box[2] > 0 else 0.
            bottom_right_y = round(float(box[3]), 2) if box[3] > 0 else 0.

            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        return prediction
