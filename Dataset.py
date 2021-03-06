import os
import glob
import random
import math

import numpy as np
from scipy import misc
from scipy import ndimage
import cv2

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def angle(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(352,160))
    # show_img = img.copy()
    rows, cols = img.shape[:2]
    img = img[85:rows,0:cols]


    mask_global = np.where((img == 1) | (img == 4)| (img == 0), 0, 1).astype('uint8')
    # kernel = np.ones((1, 1), np.uint8)
    # mask_global = cv2.erode(mask_global, kernel, iterations=2)
    mask_global = cv2.medianBlur(mask_global, 3, 4)
    contours, hierarchy = cv2.findContours(mask_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sh_cont = np.shape(contours)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    if sh_cont[0]==0 or np.shape(sh_cont)[0]>3:
        cha = 0
        # cv2.imshow("img_no", show_img*50)
        # cv2.imshow("mask_global", mask_global * 50)
        # cv2.waitKey(0)
        return cha
    else:
        rows,cols = img.shape[:2]
        img_back = np.zeros(np.shape(img))
        angle = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.uint(box)
            mask_global = cv2.drawContours(mask_global*10, [box], 0, (255,255,255), 2)
            angle.append(rect[2])

        cha = -angle[0]-(90+angle[1])
        # cv2.imshow("img",mask_global)
        # cv2.waitKey(0)
        return cha
        # cv2.imshow("img",img_back)

class Dataset:
    def __init__(self, data_dir, mode_folder):
        self.file_paths, self.num_files = TrainDataset.get_file_paths(data_dir, mode_folder)
        self.cur_index = -1  # The next image to be read would be cur_index + 1.
        self.down_size_rate = 32
        print("There are %d files in total." % self.num_files)

    @staticmethod
    def get_file_paths(data_dir, mode_folder):
        """
        Given the directory to get the path to images and annotations.
        :param data_dir: The directory to the dataset.
        :param mode_folder: The folder of images, related to mode.
        :return: Path to images and annotations.
        """
        # Define file paths list to return.
        file_paths = {
            "image": [],
            "annotation": []
        }

        # Define path for images and annotations.
        image_dir = os.path.join(data_dir, "images", mode_folder)
        annotation_dir = os.path.join(data_dir, "annotations", mode_folder)
        print("image directory:", image_dir)
        print("annotation directory:", annotation_dir)

        # For each .jpg image, find its annotation.
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        for index, image_path in enumerate(image_paths):
            # Get .png annotation path.
            full_filename = image_path.split("/")[-1]
            filename = os.path.splitext(full_filename)[0]
            annotation_path = os.path.join(annotation_dir, filename + ".png")

            # If both the image and annotation exists, append the file paths.
            if os.path.exists(annotation_path):
                file_paths["image"].append(image_path)
                file_paths["annotation"].append(annotation_path)
            else:
                print("Annotation file not found for %s - Skipping" % filename)

        return file_paths, len(file_paths["image"])

    def read_data(self, index):
        """
        Given the index, read an image and its annotation.
        :return: image: numpy.ndarray, annotation: numpy.ndarray
        """
        # Define data path.
        image_path = self.file_paths["image"][index]
        annotation_path = self.file_paths["annotation"][index]

        # Read image and annotation.
        image = misc.imread(image_path)
        annotation = misc.imread(annotation_path)
        offset = angle(annotation)
        return image, annotation, offset

    @staticmethod
    def is_data_valid(image, annotation):
        """
        Check if the image and annotation are valid.
        :param image: numpy.ndarray.
        :param annotation: numpy.ndarray.
        :return: True if image has 3 dimensions,
            annotation has 2 dimensions and they have the same size.
        """

        # 1. Check if data has correct dimensions.
        # Image must have 3 dimensions.
        # Annotation must have 2 dimensions.
        if len(image.shape) != 3 or len(annotation.shape) != 2:
            return False

        # 2. Check if image and annotation have the same size.
        # Define image and annotation's height and width.
        image_height = image.shape[0]
        image_width = image.shape[1]
        annotation_height = annotation.shape[0]
        annotation_width = annotation.shape[1]

        # Image and annotation must have the same size.
        if image_height != annotation_height or \
                image_width != annotation_width:
            return False

        return True

    def round_size(self, size):
        resize_size = math.ceil(size / self.down_size_rate) * self.down_size_rate
        max_size = 768
        if resize_size >= max_size:
            return max_size
        return resize_size

    def round_data(self, image, annotation):
        """
        :return: Resized image and annotation whose size
            can be divided by the down_size_rate.
        """
        # Define image height and width.
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Resize if height or width can't be divided with down size rate

        if image_height % self.down_size_rate != 0 or image_width % self.down_size_rate != 0:
            resize_height = math.ceil(image_height / self.down_size_rate) * self.down_size_rate
            resize_width = math.ceil(image_width / self.down_size_rate) * self.down_size_rate
            image = misc.imresize(image, [resize_height, resize_width], interp="bilinear")
            annotation = misc.imresize(annotation, [resize_height, resize_width], interp="nearest")

        return image, annotation

    def next_image(self, round_size=True, is_batch=True):
        """
        Return an image and its annotation.
        :param is_batch: If the returned data is a batch.
        :param round_size: if the size of image and its annotation should be rounded.
        :return:
            if is_batch:
                image: numpy.ndarray (1, height, width, 3),
                annotation: numpy.ndarray (1, height, width, 1)
            else:
                image: numpy.ndarray (height, width, 3),
                annotation: numpy.ndarray (height, width)
        """
        # Keep choosing an image/annotation, until the image and
        # annotation meet the requirements.
        while True:
            # Update current index.
            self.cur_index = random.randint(0, self.num_files)
            if self.cur_index >= self.num_files:
                self.cur_index = 0

            # Read the image and annotation.
            image, annotation,offset = self.read_data(self.cur_index)

            # Check if data is valid
            if not Dataset.is_data_valid(image, annotation):
                continue

            # Round the size of the image and annotation.
            if round_size:
                image, annotation = self.round_data(image, annotation)

            # Return data.
            if is_batch:
                return np.array([image]), np.expand_dims([annotation], axis=3), np.array([offset])
            else:
                return image, annotation,offset

    def get_batch(self, batch_size, is_random=False):
        """
        Get {batch_size} groups of image and annotation.
        The groups are resized to the same size.
        :param batch_size: int. The size(length) of the batch.
        :return: dict. {batch_size} groups of image and annotation
        """
        if is_random:
            # Init random start index of the batch.
            self.cur_index = random.randint(0, self.num_files - batch_size)

        # Init the batch.
        batch = {
            "image": [],
            "annotation": [],
            "offset": []
        }

        # Append {batch_size} groups of data to group.
        for i in range(batch_size):
            # Get next valid image and annotation.
            image, annotation,offset_data = self.next_image(round_size=False, is_batch=False)
            offset_data = offset_data + 90
            if  offset_data < 90 and offset_data>=80:
                offset = 3
            elif offset_data < 80 and offset_data >= 50 :
                offset = 2
            elif offset_data < 50:
                offset = 1
            elif offset_data > 90 and offset_data <= 100:
                offset = 4
            elif offset_data > 100 and offset_data <= 130:
                offset = 5
            elif offset_data > 130:
                offset = 6
            else:
                offset = 0



            # if offset_data > 87.5 and offset_data < 92.5 and offset_data!=90:
            #     offset = 5
            # elif offset_data > 80 and offset_data < 87.5:
            #     offset = 4
            # elif offset_data > 70 and offset_data < 80:
            #     offset = 3
            # elif offset_data > 55 and offset_data < 70:
            #     offset = 2
            # elif offset_data > 35 and offset_data < 55:
            #     offset = 1
            # elif offset_data < 35:
            #     offset = 0
            # elif offset_data > 92.5 and offset_data < 100:
            #     offset = 6
            # elif offset_data > 100 and offset_data < 110:
            #     offset = 7
            # elif offset_data > 110 and offset_data < 125:
            #     offset = 8
            # elif offset_data > 125 and offset_data < 145:
            #     offset = 9
            # elif offset_data > 145 :
            #     offset = 10
            # elif offset_data == 90 :
            #     offset = 11
            # else:
            #     offset = 5
            # if i>batch_size-10:
            #     model = random.randint(0, 4)
            #     # print('data')
            #     if model == 0:
            #         image = rotate(image, 2)
            #         annotation = rotate(annotation, 2)
            #         offset = angle(annotation)
            #     elif model == 1:
            #         image = rotate(image, -2)
            #         annotation = rotate(annotation, -2)
            #         offset = angle(annotation)
            #     elif model == 2:
            #         image = rotate(image, -3)
            #         annotation = rotate(annotation, -3)
            #         offset = angle(annotation)
            #     elif model == 3:
            #         image = rotate(image, 3)
            #         annotation = rotate(annotation, 3)
            #         offset = angle(annotation)


            # Append image and annotation to batch
            batch["image"].append(image)
            batch["annotation"].append(annotation)
            batch["offset"].append(offset)

        # Get max height and width of the batch
        max_height = max([image.shape[0] for image in batch["image"]])
        max_width = max([image.shape[1] for image in batch["image"]])

        # Define batch height and width.
        batch_height = self.round_size(max_height)
        batch_width = self.round_size(max_width)

        # Resize the batch to the same size.
        batch_length = len(batch["image"])
        for i in range(batch_length):
            image = batch["image"][i]
            annotation = batch["annotation"][i]

            # print(annotation.shape, batch_height, batch_width)

            batch["image"][i] = misc.imresize(image, [batch_height, batch_width], interp="bilinear")
            batch["annotation"][i] = misc.imresize(annotation, [batch_height, batch_width], interp="nearest")
            batch["annotation"][i] = np.expand_dims(batch["annotation"][i], axis=3)

        # Convert to numpy array
        batch["image"] = np.array(batch["image"])
        batch["annotation"] = np.array(batch["annotation"])
        batch["offset"] = np.array(batch["offset"])

        # Return batch.
        return batch["image"], batch["annotation"], batch["offset"]


class TrainDataset(Dataset):
    def __init__(self, data_dir):
        Dataset.__init__(self, data_dir, "training")


class TestDataset(Dataset):
    def __init__(self, data_dir):
        Dataset.__init__(self, data_dir, "validation")


class ImageReader:
    def __init__(self, image_dir):
        self.image_paths, self.num_files = ImageReader.get_image_paths(image_dir)
        self.cur_index = -1
        print("There are %d images in total." % self.num_files)

    @staticmethod
    def get_image_paths(image_dir):
        """
        Given the directory, get the path to images.
        :param image_dir: The path to the images.
        :return: Path to images.
        """
        # Define image paths list to return.
        file_paths = []

        # For each .jpg image, find its annotation.
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

        return image_paths, len(image_paths)

    def next_image(self):
        """
        Return an image.
        :return:
            image: numpy.ndarray (1, height, width, 3)
        """
        # Keep choosing an image, until all images have been read.
        while True:
            # Update current index.
            self.cur_index += 1
            if self.cur_index >= self.num_files:
                return None

            # Randomly choose an image.
            index = self.cur_index
            image_path = self.image_paths[index]

            # Read image.
            image = misc.imread(image_path)

            # Image must have 3 dimensions.
            if len(image.shape) != 3:
                continue

            # Define image's height and width.
            image_height = image.shape[0]
            image_width = image.shape[1]

            # Resize if height or width can't be divided with down size rate
            down_size_rate = 32
            if image_height % down_size_rate != 0 or image_width % down_size_rate != 0:
                resize_height = math.ceil(image_height / down_size_rate) * down_size_rate
                resize_width = math.ceil(image_width / down_size_rate) * down_size_rate
                image = misc.imresize(image, [resize_height, resize_width], interp="bilinear")

            return np.array([image])

    @property
    def has_next(self):
        return self.cur_index < self.num_files - 1
