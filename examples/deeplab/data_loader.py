'''
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np
from scipy import ndimage

np.set_printoptions(threshold=np.inf)

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


def load_images(images_list, data_folder=None, crop_size=0, start=0, batch_size=20):
    
    images = np.zeros((batch_size, 3, crop_size, crop_size))
    labels = np.zeros((batch_size, 1, crop_size, crop_size))

    for i in range(batch_size):
        line = images_list[start+i]
        img = ndimage.imread(data_folder + line.split()[0])
        if len(line.split()) == 2:
            label = ndimage.imread(data_folder + line.split()[1], flatten=True)

        height, width, channels = img.shape
        h_off = np.random.randint(height - crop_size + 1)
        w_off = np.random.randint(width - crop_size + 1)

        #mirror = np.random.randint(2)
        mirror = 0
        #crop and flip (mirror) image based on random value
        if mirror == 1:
            crop_img = np.fliplr(img[h_off:h_off+crop_size, w_off:w_off+crop_size])
            if len(line.split()) == 2:    
                crop_label = np.fliplr(label[h_off:h_off+crop_size, w_off:w_off+crop_size])
        else:
            crop_img = img[h_off:h_off+crop_size, w_off:w_off+crop_size]
            if len(line.split()) == 2:    
                crop_label = label[h_off:h_off+crop_size, w_off:w_off+crop_size]


        images[i, :, :, :] = np.transpose(crop_img)
        if len(line.split()) == 2:
            labels[i, :, :, :] = np.transpose(crop_label)

    return (images, labels)


def save_preprocessed_images(images, labels):
    
        image_file = "tmp/img.npy"
        label_file = "tmp/label.npy"
        np.save(image_file, images, allow_pickle=False)
        np.save(label_file, labels, allow_pickle=False)

def load_preprocessed_images(image_file, label_file):

    images = np.load(image_file, allow_pickle=False)
    labels = np.load(label_file, allow_pickle=False)

    return (images, labels)


'''
training_images_list = load_data(dataset="training", path="CityScapes/list/")
size = len(training_images_list)
train_data, train_label = load_images(training_images_list, data_folder="./data/", crop_size=306, start=0, batch_size=1)
print(train_data[0])

save_preprocessed_images(train_data, train_label)
images, labels = load_preprocessed_images("preprocessed_data/img.npy", "preprocessed_data/label.npy", batch_size=size, crop_size=306)
print(train_data.shape)
print(train_label.shape)
'''


