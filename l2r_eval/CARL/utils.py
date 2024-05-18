import itk
import numpy as np

class Logger:
    def __init__(self, file):
        self.file = open(file, "w")
    
    def log(self, *val):
        print(*val)
        print(*val, file=self.file)
    
    def close(self):
        self.file.close()

def itk_mean_dice(im1, im2):
    array1 = itk.array_from_image(im1)
    array2 = itk.array_from_image(im2)
    dices = []
    for index in range(1, max(np.max(array1), np.max(array2)) + 1):
        m1 = array1 == index
        m2 = array2 == index
        
        intersection = np.logical_and(m1, m2)
        
        d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
        dices.append(d)
    return np.mean(dices)

def mean_dice(im1, im2):
    array1 = im1.copy()
    array2 = im2.copy()
    dices = []
    for index in range(1, max(np.max(array1), np.max(array2)) + 1):
        m1 = array1 == index
        m2 = array2 == index
        
        intersection = np.logical_and(m1, m2)
        
        d = 2 * np.sum(intersection) / (np.sum(m1) + np.sum(m2))
        dices.append(d)
    return np.mean(dices)