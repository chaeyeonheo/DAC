# CNU Close-CV, Drone detection and Classification project(DaC)

# library import(yolo default)
from datetime import datetime

import argparse
import os
import platform
import sys
from pathlib import Path
import natsort
import math

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# For VGG11 model
import imp
import numpy as np
from PIL import Image

# For threading module
import threading
import time
import logging




#### Sub Thread execution
def work(token_folder, save_dir):
    # direction function
    def direction(x, y):
        global output # return
        O = [0, 0]
        p1 = [1, 0]
        p2 = [x, y]
        
        # vector 
        v0 = np.array(p1) - np.array(O)
        v1 = np.array(p2) - np.array(O)
        
        # angle calculation
        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        angle = np.degrees(angle)
        
        # 8-case direction
        if angle >= -22 and angle < 23:
            output = "East"
        elif angle >= 23 and angle < 68:
            output = "North-East"
        elif angle >= 68 and angle < 113:
            output = "North"
        elif angle >= 113 and angle < 158:
            output = "North-West"
        elif angle >= 158 and angle < -157:
            output = "West"
        elif angle >= -157 and angle < -112:
            output = "South-West"
        elif angle >= -112 and angle < -67:
            output = "South"
        elif angle >= -67 and angle < -22:
            output = "South-East"
            
        return output # return

    
    # create a folder to store the results
    if not os.path.isdir(str(save_dir)+"/"+"result"):
        os.makedirs(str(save_dir)+"/"+"result")    
    
    # temp list...(information)
    big_drones = []
    label_list = []

    # add information to the list
    for file in token_folder:
        f3 = open('./'+str(save_dir)+"/"+"labels/"+str(file), 'r')
        f3 = f3.readline().split(" ")
        f3 = list(map(float, f3))
        label_list.append(f3)
        big_drones.append(f3[3])

    # max_w 
    max_w= max(big_drones)

    # max_w index
    max_w_idx = big_drones.index(max_w)
    
    blue = 255 # first frame color
    red = 4  # last frame color
    
    # declare variables for direction functions
    temp = token_folder[-2]
    temp_name = temp.replace(".txt", ".jpg")

    # diretion line... target image
    dir_img = cv2.imread('./'+str(save_dir)+'/images/'+str(temp_name), cv2.IMREAD_COLOR)
    h, w, l = dir_img.shape # save
    
    # draw lines on the image
    for idx in range(len(label_list)-1):
        cv2.line(dir_img, (int(label_list[idx][1]*w), int(label_list[idx][2]*h)),
                             (int(label_list[idx+1][1]*w), int(label_list[idx+1][2]*h)), (blue, 0, red), 3)
        red += 4
        blue -= 4
    
    # point calculation for direction calculation
    p2_x = label_list[9][1] - label_list[0][1]
    p2_z = (label_list[9][3]/max_w) - (label_list[0][3]/max_w) # ACHTUNG! not an absolute value!!
    
    # direction function 
    dir_output = direction(p2_x, p2_z) # x, y(X) x, z(O)
    
    # Target img!!(for VGG11)
    ddrone = str(token_folder[max_w_idx])[:-4]+".jpg"
    
    
    
    ### Drone classification
    # model fitting
    vgg11 = torch.jit.load('vgg11.pth')
    vgg11.eval()

    # image processing
    target_data = [] # image information list
    target_img = Image.open(str(save_dir)+"/crops/"+str(ddrone)) # open image
    resize_target_img = target_img.resize((224, 224)) # resize
    r, g, b = resize_target_img.split() # channel split(r, g, b)
    
    # nomalization(0 ~ 1)
    r_resize_img = np.asarray(np.float32(r) / 255.0)
    b_resize_img = np.asarray(np.float32(g) / 255.0)
    g_resize_img = np.asarray(np.float32(b) / 255.0)

    # pasting
    rgb_target_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])
    target_data.append(rgb_target_img)
    target_data = np.array(target_data, dtype='float32')
    target_data = torch.from_numpy(target_data).float() # 잘 사용할 수 있는 form으로 최종 설정

    # classes
    classes =  ('DJI_Air2S', 'DJI_FPVcomdo', 'DJI_Mini2', 'DJI_Phatom4Pro', 'DJI_Tello', 'DJI_Phantom3')

    correct = 0 
    total = 0 

    with torch.no_grad():
        vgg11.eval() # model
        images = target_data.cuda()
        outputs = vgg11(images).cuda() 
        predicted = torch.argmax(outputs)
        
        # save results
        with open(str(save_dir)+"/result/time_stamp.txt", 'a') as f:
            f.write(str(token_folder[0])+" ==> "+str(classes[predicted])+'\n')
            
    # SAVE(direction, class, time information)
    string = str(dir_output+"  "+str(classes[predicted]) +"  "+ str(temp[:-4]))
    
    # draw white box
    dir_img = cv2.rectangle(dir_img, (0, 460, 640, 40), (255, 255, 255), -1)
    # enter text....
    dir_img = cv2.putText(dir_img, string, (0, 475), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,0))
    # image...save!
    cv2.imwrite('./'+str(save_dir)+"/"+"result/"+str(temp[:-8])+"_2D_direction.jpg", dir_img)

    
    # One more time..! (for images to video)
    blue = 255
    red = 4
    
    img_array = []
    
    # accessing list elements
    for idx in range(len(label_list)-1):
        file_info = token_folder[idx]
        frame = cv2.imread('./'+str(save_dir)+'/images/'+file_info[:-4]+'.jpg')
        h, w, l = frame.shape
        size = (w, h)
        
        # draw line
        cv2.line(frame, (int(label_list[idx][1]*w), int(label_list[idx][2]*h)),
                             (int(label_list[idx+1][1]*w), int(label_list[idx+1][2]*h)), (blue, 0, red), 3)
        red += 4
        blue -= 4
        
        # draw rectangle, text... and save frame information
        frame = cv2.rectangle(frame, (0, 460, 640, 40), (255, 255, 255), -1)
        frame = cv2.putText(frame, string, (0, 475), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,0))
        img_array.append(frame)
        
        # save! video!!!!!!!
    out=cv2.VideoWriter('./'+str(save_dir)+'/result/'+temp[:-8]+'_project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

    # information write
    for i in range(len(img_array)):
        out.write(img_array[i])
#     out.relaease()
            
    try:
        # delete unnecessary files
        for file in token_folder:
            file = file[:-4]
            os.remove("./"+str(save_dir)+"/images/"+str(file)+".jpg")
            os.remove("./"+str(save_dir)+"/crops/"+str(file)+".jpg")
            os.remove("./"+str(save_dir)+"/labels/"+str(file)+".txt")
    except:
        pass

    print("WOW! SAVE OUTPUT! DRONE DETECTOR, DaC!") # finally...!
    

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s" # Logging format
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    
    @torch.no_grad()
    def run(
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):

        global line
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
        (save_dir / 'images').mkdir(parents=True, exist_ok=True)

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path

                # Store information about the current time
                now = datetime.now()
                now = str(now).replace(" ", "_")
#                 now = now.replace(" ", "_")
                now = now[:19]

                # Save path and file name with current time
                save_path = str(save_dir / 'images' / now) + f'_{frame}' + '.jpg'
                txt_path = str(save_dir / 'labels' / now) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

                # Drone detection -> False to True
                save_drone = False

                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
    #                     if save_txt:  # Write to file
                        if cls == 0: # drone
                            save_drone = True
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (1, *xywh)

                            file_path = "./"+str(save_dir)+"/labels/"
                            token_folder = os.listdir(file_path) # folder token

                            if (len(token_folder)) > 0: # works if folder is not empty
                                stay = []
                                for file in token_folder:
                                    stay.append(int(file[20:-4]))
                                stay = list(map(int, stay))
                                stay_list = list(map(lambda x : x - frame, stay))
                                stay_result = max(stay_list)
                                stay_idx = stay_list.index(stay_result)
                                
                                # f1 and f2 save
                                f1 = open(str(file_path)+str(token_folder[stay_idx]), 'r')
                                f1 = f1.readline().split(" ")
                                f1 = list(map(float, f1))
                                f2 = line

                                try:
                                    # Calculation of change
                                    change_x = abs((f1[1] - f2[1]) / max(f1[3], f2[3]))
                                    change_y = abs((f1[2] - f2[2]) / max(f1[4], f2[4]))
                                    max_info = change_y / change_x
                                except:
                                    max_info = 0
                                
                                if max_info > 10: # 10 == 𝜎, delete
                                    os.remove("./"+str(file_path)+str(token_folder[stay_idx]))
                                    os.remove("./"+str(save_dir)+"/images/"+str(token_folder[stay_idx])[:-4] + ".jpg")
                                    os.remove("./"+str(save_dir)+"/crops/"+str(token_folder[stay_idx])[:-4] + ".jpg")
                            
                            token_folder = os.listdir(file_path) # folder token
                            token_folder = natsort.natsorted(token_folder) # sorting
                            

                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                    
                            if len(token_folder) == 61:
                                args_thread = token_folder[:-2]
                                    
                                x = threading.Thread(target=work, args=(args_thread, save_dir)) 
                                x.start() # sub thread start!


                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if save_crop: # file name with current time
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{now}_{frame}.jpg', BGR=True)

                # Stream results
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_drone: # drone!
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)

                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 1, im0.shape[1], im0.shape[0]
                            cv2.imwrite(save_path, im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s) {frame}')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



    # Parse!
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt


    def main(opt):
        check_requirements(exclude=('tensorboard', 'thop'))
        run(**vars(opt))


    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
