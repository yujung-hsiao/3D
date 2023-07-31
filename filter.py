import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
import os

def MakeVideo(img_files, fps, video_name):
    if isinstance(img_files[0], str):
        frame = []
        for img in img_files:
            frame.append(cv2.imread(img))
        clip = ImageSequenceClip(frame, fps)
    else:
        clip = ImageSequenceClip(img_files, fps)
    clip.write_videofile(video_name)

# color = ['white', 'blue']
# fps = [30, 60, 90]
# tmp = 0
# # bg = cv2.imread('0723/bg_auto_t1.png', 0)
# frame = []
# for c in color:
#     folder = f'0723/{c}/test1'
#     for f in fps:
#         cnt = 0
#         for i in range(1, (f+1)):
#             name = str(i)
#             if i < 10:
#                 name = f'0{i}'
#             img = cv2.imread(f'{folder}/{c[0]}_f{f}_t1_{name}.png')
#             sub = cv2.subtract(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bg)
#             a = np.mean(sub.ravel())
#             if a < 1:
#                 cnt += 1
#                 continue

#             # frame.append(img)
#         # MakeVideo(frame, 20, f'0723/{c[0]}_f{f}_test3.mp4')

#         print(f'no pattern: {cnt}')


# size = (800, 1280)
# color = ['white', 'blue']
# fps = [30, 60, 90]

# for c in color:
#     folder = f'0723/{c}/b0'
#     for f in fps:
#         img_list = [os.path.join(folder, img)
#                     for img in sorted(os.listdir(folder))
#                     if img.startswith(f'{c[0]}_f{f}')]
#         print(c, f, len(img_list))
#         MakeVideo(img_list, 20, f'0723/{c[0]}_f{f}_test3.mp4')

img_file = [os.path.join('0725', img)
         for img in sorted(os.listdir('0725'))
         if img.endswith('.jpg')]

MakeVideo(img_file, 20, 't1.mp4')

frame = [cv2.imread(i) for i in img_file]
MakeVideo(frame, 10, 't2.mp4')