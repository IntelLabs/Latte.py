import numpy as np
import cv2



def load_deeplab(dataset="training", path=None, data_folder=None, is_color=True, crop_size=0):

    if path is None or data_folder is None:
           raise ValueError("Unspecified path or data folder. Unable to load data!")
    
    if dataset == "training":
        path += "trainval.txt"
    else:
        path += "test.txt"

    images_list = open(path, 'r')
    lines = images_list.readlines()
    images = []
    labels = []

    for line in lines:
        img = cv2.imread(data_folder + line.split()[0], is_color)
        if len(line) == 2:
            label = cv2.imread(data_folder + line.split()[1], is_color)

        height, width, channels = img.shape
        h_off = np.random.randint(height - crop_size + 1)
        w_off = np.random.randint(width - crop_size + 1)

        mirror = np.random.randint(2)

        #crop and flip (mirror) image based on random value
        if mirror == 1:
            crop_img = cv2.flip(img[h_off:h_off+crop_size, w_off:w_off+crop_size], 1)
            if len(line) == 2:    
                crop_label = cv2.flip(label[h_off:h_off+crop_size, w_off:w_off+crop_size], 1)
        else:
            crop_img = img[h_off:h_off+crop_size, w_off:w_off+crop_size]
            if len(line) == 2:    
                crop_label = label[h_off:h_off+crop_size, w_off:w_off+crop_size]

        images.append(crop_img)
        if len(line) == 2:
            labels.append(crop_label)

    return (images, labels)



src = "CityScapes/list/"
data_folder = "data/"
images, labels = load_deeplab("training", src, data_folder, True, 306)
