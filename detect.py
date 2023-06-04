
import argparse
import os
import sys
import json
from pathlib import Path
from time import time
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_img_size, check_suffix, colorstr,
                           increment_path, non_max_suppression, print_args,scale_coords, xyxy2n, n2xyxy)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from mmdet.apis import init_detector, inference_detector
from ensemble_boxes import weighted_boxes_fusion
from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer
from importlib import import_module
from toJson.jsonResult import jsonResult

import numpy as np

@torch.no_grad()
def run(weights, source, imgsz, conf_thres, iou_thres, max_det, device,
        save_txt, nosave, classes, project, name, exist_ok, line_thickness
        ):
    n_config_file = '/home/akay/tph-yolov5/mm/visdrone_cnext_fpn_thead/visdrone_cnext_fpn_thead.py'
    n_checkpoint_file = '/home/akay/tph-yolov5/mm/visdrone_cnext_fpn_thead/best_bbox_mAP_epoch_12.pth'
    
    p_config_file= 'mm/pistv3_cnext_fpn_thead/pistv3_cnext_fpn_thead.py'
    p_checkpoint_file= 'weights/pistv3.pth'
    
    padim_config= get_configurable_parameters(config_path="anomalib/models/padim/config.yaml")
    padim_module = import_module("anomalib.deploy.inferencers.torch")
    TorchInferencer = getattr(padim_module, "TorchInferencer")  
    uai_inferencer = TorchInferencer(config=padim_config, model_source="weights/uai_padim.ckpt")
    uap_inferencer= TorchInferencer(config=padim_config, model_source="weights/uap_padim.ckpt")
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)

    # Load model
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    #n_model = init_detector(n_config_file, n_checkpoint_file, device='cuda:0')
    p_model= init_detector(p_config_file,p_checkpoint_file,device='cuda:0')
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    names= ['insan', 'insan', 'tasit', 'tasit', 'tasit', 'tasit',
                   'tasit', 'tasit', 'tasit', 'tasit','uap','uai']
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader  
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=".pt")
    
    # Run inference
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    enum=0
    for path, img, im0s, vid_cap, s in dataset: 
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=False, visualize=False)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)

        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1 
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            height,width,channel=im0s.shape
            
            # mm-object segment
            #n_result = inference_detector(n_model, im0s)
            mmo_threshold=0.6
            mmo_boxes=[]
            mmo_confs=[]
            mmo_labels=[]
            # for e,mmo in enumerate(n_result):
            #     if len(mmo)>0:
            #         for modet in mmo:
            #             if modet[-1]>=mmo_threshold:
            #                 box= list(map(lambda x:round(float(x),2) if x>0 else 0,modet[:4]))
            #                 mmo_boxes.append(xyxy2n([round(x,3) for x in modet[:4]],width,height))
            #                 mmo_confs.append(round(modet[-1],3))
            #                 mmo_labels.append(e)
            tpho_boxes=[]
            tpho_confs=[]
            tpho_labels=[]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                
                for tpho in det:
                    tpho= tpho.cpu().numpy()
                    box= list(map(lambda x:round(float(x),2) if x>0 else 0,tpho[:4]))
                    tpho_boxes.append(xyxy2n(box,width,height))
                    tpho_confs.append(round(float(tpho[4]),2))
                    tpho_labels.append(int(tpho[-1]))
        
            boxes_list= [mmo_boxes,tpho_boxes]
            scores_list= [mmo_confs,tpho_confs]
            labels_list= [mmo_labels,tpho_labels]
            weights_list= [0,5]
            boxes_n, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                          weights=weights_list, iou_thr=mmo_threshold, skip_box_thr=0.25)
            
            boxes=[]
            scores= scores.tolist()
            labels= labels.astype(np.int8).tolist()
            if len(boxes_n):
                for e,mbox in enumerate(boxes_n):
                    mbox= n2xyxy(mbox,width,height)
                    boxes.append(mbox) 
            
            pist_score_list=["-1" for x in range(len(boxes))]
            p_result= inference_detector(p_model,im0s)
            mmp_threshold= 0.89
            for e,mmp in enumerate(p_result):
                if len(mmp)>0:
                    for mpdet in mmp:
                        if mpdet[-1]>=mmp_threshold:
                            
                            box=list(map(lambda x:round(x,2) if x>0 else 0,mpdet[:4]))
                            boxes.append(box)
                            scores.append(round(mpdet[-1],3))
                            labels.append(e+10)
                            
                            padim_img= im0s[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
                            padim_img= cv2.resize(padim_img,(224,224))
                            
                            if e==0:
                                padim_output = uap_inferencer.predict(image=padim_img, superimpose=False, overlay_mask=False)
                                if isinstance(padim_output, tuple):
                                    anomaly_map, padim_score = padim_output
                                    
                            elif e==1:
                                padim_output = uai_inferencer.predict(image=padim_img, superimpose=False, overlay_mask=False)
                                if isinstance(padim_output, tuple):
                                    anomaly_map, padim_score = padim_output
                                    
                            if padim_score>=0.75:
                                pist_score_list.append("0")
                            elif padim_score<0.75:
                                pist_score_list.append("1")                                      
            
            if len(boxes)>0:
                for xyxy, conf, cls in zip(boxes, scores, labels):
                    
                    if conf> 0.4:

                        if save_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = (f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
            
            json_res= jsonResult(boxes,labels,pist_score_list,enum,path)
            
            enum+=1
            if not os.path.exists(f"{str(save_dir)}/json_files"):
                os.makedirs(f"{str(save_dir)}/json_files")
                
            json_name= path.rsplit("/")[-1].rsplit(".")[0]
            json_save_file= open(f"{str(save_dir)}/json_files/{json_name}.json","w")
            json.dump(json_res,json_save_file,indent=4)  
            json_save_file.close()   

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            #Stream results
            im0 = annotator.result()

           # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5l-xs-2.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='./datasets/oturum1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1536], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.40, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', default= True, action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    start_time= time()
    run(**vars(opt))
    end_time= time()
    print(f"***************     elapsed time: {end_time-start_time:.2f} seconds.    ***************")
    


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
