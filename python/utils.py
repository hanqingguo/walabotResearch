from skimage import transform
import os
import numpy as np
from random import shuffle

class Rescale(object):
    """
       Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        #img = transform.resize(image, (new_h, new_w), mode='constant', anti_aliasing=True)
        img = transform.resize(image, (new_h, new_w), mode='constant')
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return img

class ToTensor(object):

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2,0,1))
        return image


def select_video(current_dir, training=True):
    """

    :param current_dir: the path upper training
    :return: class name of the video, and the real path of the video
    """
    if(training):
        video_dir = os.path.join(os.path.dirname(current_dir), 'training')
    else:
        video_dir = os.path.join(os.path.dirname(current_dir), 'testing')
    classes = os.listdir(video_dir)
    random_class = np.random.randint(len(classes)) # get random class folder
    classname = classes[random_class]
    video_list = os.listdir(os.path.join(video_dir,classname))
    random_video = np.random.randint(len(video_list))
    selected_video = os.path.join(video_dir, classname, video_list[random_video])


    return classname, selected_video

def video_loader(current_dir, training=True):
    """

    :param current_dir: the path upper training
    :param training: True if training
    :return: list
            ['walk 4.avi', 'walk 3.avi', 'stand-to-sit 21.avi', 'stand-to-sit 25.avi', ..., 'walk 18.avi', 'jump 24.avi']
    """
    if(training):
        video_dir = os.path.join(os.path.dirname(current_dir), 'test_dataset')
    else:
        video_dir = os.path.join(os.path.dirname(current_dir), 'testing')


    classes = os.listdir(video_dir)
    ordered_class_list = []

    for cls in classes:
        #if cls not in random_order_dict:
        video_list = os.listdir(os.path.join(video_dir, cls))
        for video in video_list:
            video = cls+" "+video
            ordered_class_list.append(video)
    shuffle(ordered_class_list)

    return ordered_class_list

current_dir = os.path.dirname(os.path.realpath(__file__))
random_order_list = video_loader(current_dir)



video_dir = os.path.join(os.path.dirname(current_dir), 'test_dataset')

# Iter all dataset
for value in random_order_list:
    [cls, video] = value.split()
    video_path = os.path.join(video_dir, cls, video)

