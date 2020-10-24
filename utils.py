#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:24:42 2020

@author: roshan
"""
import openslide
import math
import tensorflow as tf
import numpy as np
import skimage.morphology as sk_morphology
import PIL

SCALE_FACTOR = 64

def slide_to_scaled_pil_image(path):
        slide = openslide.open_slide(path)
        large_w, large_h = slide.dimensions
        new_w = math.floor(large_w / SCALE_FACTOR)
        new_h = math.floor(large_h / SCALE_FACTOR)
        level = slide.get_best_level_for_downsample(SCALE_FACTOR)
        whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        whole_slide_image = whole_slide_image.convert("RGB")
        img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        img = np.array(img)
        return img

def mask_percent(np_img):
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - tf.math.count_nonzero(np_sum,dtype=tf.int32) / tf.size(np_sum) * 100
    else:
        mask_percentage = 100 - tf.math.count_nonzero(np_img,dtype=tf.int32) / tf.size(np_img) * 100
    return mask_percentage

def filter_grays(rgb, tolerance=15):
    (h, w, c) = rgb.shape
    rgb = tf.cast(rgb,tf.int32)
    rg_diff = tf.math.abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = tf.math.abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = tf.math.abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)
    return result

def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh):
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    return result


def filter_red_pen(rgb):
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
               filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
               filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
               filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
               filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
               filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
               filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
               filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
               filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    return result

def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh):
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result


def filter_green_pen(rgb):
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
               filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
               filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
               filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
               filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
               filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
               filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
               filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
               filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
               filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
               filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
               filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
               filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
               filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
               filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    return result

def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh):
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result

def filter_blue_pen(rgb):
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
               filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
               filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
               filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
               filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
               filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
               filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
               filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
               filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
               filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
               filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
               filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    return result

def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95):
    rem_sm = tf.cast(np_img,tf.bool) 
    rem_sm = sk_morphology.remove_small_objects(rem_sm.numpy(), min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh)
    np_img = rem_sm
    return np_img

def mask_rgb(rgb, mask):
    mask_1 = tf.stack([mask,mask,mask] , axis=-1)
    mask_1 = tf.cast(mask_1,tf.uint8)
    result = rgb*mask_1
    return result

def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90):
    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh)
    np_img = gr_ch_mask
    return np_img

def apply_image_filters(rgb):
    mask_not_green = filter_green_channel(rgb)
    
    mask_not_gray = filter_grays(rgb)

    mask_no_red_pen = filter_red_pen(rgb)

    mask_no_green_pen = filter_green_pen(rgb)

    mask_no_blue_pen = filter_blue_pen(rgb)
    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
    
    mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500)
    rgb_remove_small = mask_rgb(rgb, mask_remove_small)
    not_greenish = filter_green(rgb_remove_small, red_upper_thresh=125, green_lower_thresh=30, blue_lower_thresh=30)
    not_grayish = filter_grays(rgb_remove_small, tolerance=30)
    rgb_new = mask_rgb(rgb_remove_small, not_greenish & not_grayish)
    return rgb_new
