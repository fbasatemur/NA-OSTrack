import cv2 as cv
import glob

LaSOT_VID_DATASETS = [  #'airplane-1',
                        'airplane-2',
                        #'airplane-3',
                        'airplane-4',
                        'airplane-5',
                        #'airplane-6',
                        #'airplane-7',
                        #'airplane-8',
                        #'airplane-9',
                        #'airplane-10',
                        #'airplane-11',
                        #'airplane-12',
                        'airplane-13',
                        #'airplane-14',
                        #'airplane-15',
                        #'airplane-16',
                        'airplane-17'
                        #'airplane-18',
                        #'airplane-19'
                        ]

IMG_BASE_PATH = "/home/root/Desktop/repos/_dataset/LaSOT_customize/airplane/"
BBOX_BASE_PATH = "/home/root/Desktop/repos/_trackers/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300/"


SAVE_VIDEO_PATH = BBOX_BASE_PATH
SIZE_W, SIZE_H = 1280, 720
FRAME_NO_OFFSET = "0" * 7

def load_paths(folder, img_paths):
    filenames = [img for img in glob.glob(folder + "/img/*.jpg")]
    filenames.sort()
    img_paths += filenames


# x, y, w, h
def load_bboxes(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines() 
        lines = [x.split("\t") for x in lines]

        return lines
def load_bboxes_from_files(files, all_boxes):
    for sub_file in files:
        all_boxes += load_bboxes(BBOX_BASE_PATH + sub_file + ".txt")

def draw_boxes(imgs_paths, all_boxes, video = None):
    if video:
        sub_dataset, counter = [], 1
        str_counter = ""
        text = ""
        for i in range(len(imgs_paths)):
            if imgs_paths[i].split("/")[-3] not in sub_dataset:
                sub_dataset.append(imgs_paths[i].split("/")[-3])
                print(sub_dataset[-1])
                counter=1
            img, bbox = cv.imread(imgs_paths[i]), all_boxes[i]
            cv.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), (0, 255, 0), 1)
            str_counter = str(counter)
            text = sub_dataset[-1] + "  " + FRAME_NO_OFFSET[:-len(str_counter)] + str_counter
            video.write(cv.putText(cv.resize(img, (SIZE_W, SIZE_H)), text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0)))
            counter+=1
    video.release()


video = cv.VideoWriter(SAVE_VIDEO_PATH + f"airplane_vitb_256_mae_ce_32x4_ep300_{LaSOT_VID_DATASETS[0]}_v12_bicubic_dimpx.mp4",cv.VideoWriter_fourcc(*"mp4v"), 30, (SIZE_W, SIZE_H))
img_paths = []
all_boxes = []
for sub_folders in LaSOT_VID_DATASETS:
    load_paths(IMG_BASE_PATH + sub_folders, img_paths)
load_bboxes_from_files(LaSOT_VID_DATASETS, all_boxes)
draw_boxes(img_paths, all_boxes, video)
