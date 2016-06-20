import numpy as np
import cv2


def load_data(dataset="training", path=None):

    if path is None:
           raise ValueError("Unspecified path or data folder. Unable to load data!")
    
    if dataset == "training":
        path += "trainval.txt"
    else:
        path += "test.txt"

    images_list = open(path, 'r')
    lines = images_list.readlines()

    return lines


def load_images(images_list, data_folder=None, is_color=True, crop_size=0, start=0, batch_size=20):
    
    images = []
    labels = []

    for i in range(batch_size):
        line = images_list[start+i]
        img = cv2.imread(data_folder + line.split()[0], is_color)
        if len(line.split()) == 2:
            label = cv2.imread(data_folder + line.split()[1], is_color)

        height, width, channels = img.shape
        h_off = np.random.randint(height - crop_size + 1)
        w_off = np.random.randint(width - crop_size + 1)

        mirror = np.random.randint(2)

        #crop and flip (mirror) image based on random value
        if mirror == 1:
            crop_img = cv2.flip(img[h_off:h_off+crop_size, w_off:w_off+crop_size], 1)
            if len(line.split()) == 2:    
                crop_label = cv2.flip(label[h_off:h_off+crop_size, w_off:w_off+crop_size], 1)
        else:
            crop_img = img[h_off:h_off+crop_size, w_off:w_off+crop_size]
            if len(line.split()) == 2:    
                crop_label = label[h_off:h_off+crop_size, w_off:w_off+crop_size]

        images.append(np.transpose(crop_img))
        if len(line.split()) == 2:
            labels.append(np.transpose(crop_label))

    return (images, labels)

