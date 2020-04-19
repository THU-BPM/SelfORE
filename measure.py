from eval import *
DATASET="nyt_fb_v2"
K_NUMS = [10, 20, 30]
LABEL_NUM = 10
LOOP_NUM = 5
measure(K_NUMS, LABEL_NUM, DATASET, LOOP_NUM)
