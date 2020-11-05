#使用方法：在命令行输入python xtq_main_pic.py -i ./test_kf/ -o ./test_save -t ./00424.png -s 80
#i是存放图片的文件夹，记得加‘/’;o是输出处理后图片的文件夹；t是模板照片;s是脸部质量的分数，默认75
import argparse
import cv2
import numpy as np
import pprint
import requests
import time
from json import JSONDecoder
from skimage import transform, io
import os
import operator
from scipy.signal import argrelextrema

np.random.seed(0)


def get_args():
    parser = argparse.ArgumentParser(description="Crop an image.")
    parser.add_argument("-i", "--input_image_path", type=str, default="./input_data/test_origin_resize.jpg", help="The path of the image you want to crop.")
    parser.add_argument("-o", "--output_image_path", type=str, default="./output_data/test_origin_resize_crop.jpg", help="The result of the crop.")
    parser.add_argument("-t", "--template", type=str, default="./00424.png", help="Wether to aqcuire template landmarks. Default is true.")
    parser.add_argument("-s", "--score", type=int, default=75, help="The score of facequality.")
    return parser.parse_args()


def smooth(x, window_len=13, window='hanning'):
    print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


class Frame:
    """class to hold information about each frame

    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
    x = (b - a) / max(a, b)
    print(x)
    return x


def Get_Imgvec(videopath,save_path):
    rootdir = save_path  # 保存图片路径
    #清空原先的文件夹
    for i in os.listdir(rootdir):
        path_file = os.path.join(rootdir, i)
        os.remove(path_file)

    # Setting fixed threshold criteria
    USE_THRESH = False
    # fixed threshold value
    THRESH = 0.6
    # Setting fixed threshold criteria
    USE_TOP_ORDER = False
    # Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    # Number of top sorted frames
    NUM_TOP_FRAMES = 10

    # Directory to store the processed frames
    dir = save_path
    # smoothing window size
    len_window = int(40)

    # load video and compute diff between frames
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        prev_frame = curr_frame
        print(i)
        i = i + 1
        success, frame = cap.read()
    cap.release()

    # compute keyframe
    keyframe_id_set = set()
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("diff"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            keyframe_id_set.add(keyframe.id)
    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                keyframe_id_set.add(frames[i].id)
    if USE_LOCAL_MAXIMA:
        print("Using Local Maxima")
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        for i in frame_indexes:
            keyframe_id_set.add(frames[i - 1].id)

        #plt.figure(figsize=(40, 20))
        #plt.locator_params(numticks=100)
        #plt.stem(sm_diff_array)
        #plt.savefig(dir + 'plot.png')

    # save all keyframes as image
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    keyframes = []
    success, frame = cap.read()
    idx = 0
    num = 1
    while (success):
        if idx in keyframe_id_set:
            name = "kf_" + str(num) + ".jpg"
            cv2.imwrite(dir + name, frame)
            keyframe_id_set.remove(idx)
            num = num+1
        idx = idx + 1
        success, frame = cap.read()
    cap.release()


def get_landmarks(img_path,score):
    URL = "https://api-cn.faceplusplus.com/facepp/v3/detect"

    data = dict()
    data["api_key"] = "Oz-xhKkok8YDa12CV1k_aJAl4VOkwPFs"
    data["api_secret"] = "rDPvjaqUj_w7xg06w2BLOILKIHK2WkaN"
    data["return_landmark"] = "2"
    data["return_attributes"] = "facequality"

    files = {"image_file": open(img_path, "rb")}

    # post
    response = requests.post(url=URL, data=data, files=files)
    res_content = response.content.decode("utf-8")
    res_dict = JSONDecoder().decode(res_content)

    # get raw landmarks
    if "faces" in res_dict.keys() and len(res_dict["faces"])>0:
        if res_dict["faces"][0]["attributes"]["facequality"]["value"]>score:
            landmarks = res_dict["faces"][0]["landmark"]
            raw_landmarks = []
            for name, point in landmarks.items():
                raw_landmarks.append([point["x"], point["y"]])
            raw_landmarks = np.asarray(raw_landmarks)
            return raw_landmarks,True
        else:
            return 0,False 
    else:
        return 0,False

'''
draw landmarks on given image and save it to ./display_data.
This function is for test mainly.
img: a cv2 image.
landmarks: a numpy array shaped as (N, 2)
'''

def draw_landmarks(img, landmarks):
    num = len(landmarks)
    for i in range(num):
        cv2.circle(img, tuple(landmarks[i]), 5, (60, 20, 220), thickness=5)
    cv2.imwrite(f"./display_data/{time.time()}.jpg", img)


def crop_image(target, target_landmarks, template_landmarks):
    # get affine matrix M
    transform_instance = transform.SimilarityTransform()
    transform_instance.estimate(target_landmarks, template_landmarks)
    M = transform_instance.params[:2, :]

    # warp affine
    # target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)
    target_crop = cv2.warpAffine(target, M, (1024, 1024))
    return target_crop

if __name__ == "__main__":
    args = get_args()
    dst = args.input_image_path  #保存关键帧的文件夹
    final_fetch = args.output_image_path   #保存最后提取人脸的文件夹
    sc = args.score
    TEMPLATE_PATH = args.template
    template_landmarks,nu = get_landmarks(TEMPLATE_PATH,0)#见上面

    kf_name = os.listdir(dst)
    ji = 1
    al_num = 1
    for pic in kf_name:
        pic_path = dst+pic
        target_landmarks,judge = get_landmarks(pic_path,sc)
        print("process "+str(al_num)+"th image:",pic_path)
        if judge:
            target = cv2.imread(pic_path)
            target_crop = crop_image(target, target_landmarks, template_landmarks)
            sp = final_fetch+'/'+str(ji)+'.jpg'
            cv2.imwrite(sp, target_crop)
            ji = ji+1
        al_num = al_num+1
