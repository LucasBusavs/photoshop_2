import cv2
import numpy as np
from img_process.util import complete_img
from img_process.local_process import convolutional_mask
import math

        

def median_filter(image, filter_size=5):
    if (filter_size % 2) == 0:
        raise Exception("Size filter error! Please select a filter with odd size.")
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padding = int((filter_size-1)/2)
    
    completed_img = complete_img(gray_img, padding)
    
    final_img = []
    for index_x, x in enumerate(completed_img[padding:-padding]):
        row_list = []

        for index_y,y in enumerate(x[padding:-padding]):
            median_img = []

            for i in range(filter_size):
                for j in range(filter_size):
                    median_img.append(completed_img[index_x + i][index_y + j])

            median_img.sort()
            row_list.append(median_img[((filter_size*filter_size)+1)//2])
        final_img.append(row_list)

    # Convert final_img to a NumPy array for display
    final_img_array = np.array(final_img, dtype=np.uint8)
    
    return final_img_array

def avg_filter(image, filter_size=5):
    if (filter_size % 2) == 0:
        raise Exception("Size filter error! Please select a filter with odd size.")

    padding = int((filter_size-1)/2)
 
    mask = []
    for y in range(-padding, padding + 1):
        row_list = []

        for x in range(-padding, padding + 1):
            
            row_list.append(1/filter_size**2)
        mask.append(row_list)
        
    # Convert final_img to a NumPy array for display
    final_img_array = convolutional_mask(image, mask)
    
    return final_img_array

def laplacian_filter(image, mask_size = 3, sigma = 1.0):
    height_img, width_img, _ = image.shape
    
    if (mask_size % 2) == 0:
        raise Exception("Size filter error! Please select a filter with odd size.")
    
    if (mask_size > height_img) or (mask_size > width_img):
        raise Exception("Size filter error! Please select a filter with lower size.")
    
    padding = int((mask_size - 1) / 2)
    mask = []
    total_sum = 0.0
    
    for y in range(-padding, padding + 1):
        row_list = []
        for x in range(-padding, padding + 1):
            
            gauss_term = (x*x + y*y) / (2 * sigma*sigma)
            value = - (1 / (math.pi * math.pow(sigma, 4))) * (1 - gauss_term) * math.exp(-gauss_term)
            
            row_list.append(value)
            total_sum += value
            
        mask.append(row_list)
    
    central_value = mask[padding][padding]
    sum_nb = total_sum - central_value

    mask[padding][padding] = -sum_nb
    
    return convolutional_mask(image, mask)

def derivative_filter(image, mask_size = 3, direction = 'x'):
    height_img, width_img, _ = image.shape
    
    if (mask_size % 2) == 0:
        raise Exception("Size filter error! Please select a filter with odd size.")
    
    if (mask_size > height_img) or (mask_size > width_img):
        raise Exception("Size filter error! Please select a filter with lower size.")
    
    padding = int((mask_size - 1) / 2)
    mask = []
    
    for y in range(-padding, padding + 1):
        row_list = []
        for x in range(-padding, padding + 1):
            
            if direction.lower() == 'x':
                if x == 0:
                    value = 0.0
                else:
                    weight = 2.0 if y == 0 else 1.0
                    value = (x / abs(x)) * weight
                    
            elif direction.lower() == 'y':
                if y == 0:
                    value = 0.0
                else:
                    weight = 2.0 if x == 0 else 1.0
                    value = (y / abs(y)) * weight
            else:
                raise Exception("Wrong direction! Choose x or y.")
                
            row_list.append(value)
        mask.append(row_list)
        
    return convolutional_mask(image, mask)

def gaussian_filter(image, mask_size = 3):
    height_img, width_img, _ =  image.shape
    
    if (mask_size % 2) == 0:
        raise Exception("Size filter error! Please select a filter with odd size.")
    
    if(mask_size>height_img) or (mask_size> width_img):
        raise Exception("Size filter error! Please select a filter with lower size.")
    
    padding = int((mask_size-1)/2)
    mask = []
    for y in range(-padding, padding + 1):
        row_list = []

        for x in range(-padding, padding + 1):
            
            distance = math.sqrt(x*x + y*y)
            row_list.append(distance)
        mask.append(row_list)
    
    return convolutional_mask(image, mask)


