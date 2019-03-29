#!/usr/bin/env python

from __future__ import division

import argparse
import io
import math
import os
import random

import numpy as np
import tensorflow as tf

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, '../image-data/labels-map.csv')
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../tfrecords-output')
DEFAULT_NUM_SHARDS_TRAIN = 3
DEFAULT_NUM_SHARDS_TEST = 1


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter(object):
    """
    Class that handles converting images to TFRecords.
    이미지를 TFRecord로 변환하는 것을 관리하는 클래스
    """

    def __init__(self, labels_csv, label_file, output_dir,
                 num_shards_train, num_shards_test):

        # TFRecord 파일 저장 디렉토리
        self.output_dir = output_dir
        # 트레이닝 데이터를 몇 개의 파일로 나누어 저장할지 지정
        self.num_shards_train = num_shards_train
        # 테스트 데이터를 몇 개의 파일로 나누어 저장할지 지정
        self.num_shards_test = num_shards_test

        # TFRecord를 저장할 디렉토리가 없다면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get lists of images and labels.
        # image 경로와 label 정보를 csv파일과 txt 파일로부터 읽어들임
        self.filenames, self.labels = \
            self.process_image_labels(labels_csv, label_file)

        # Counter for total number of images processed.
        # 이미지 처리의 총 숫자를 세는 카운터 생성
        self.counter = 0

    def process_image_labels(self, labels_csv, label_file):
        """
        This will constuct two shuffled lists for images and labels.
        이렇게하면 이미지와 레이블에 대해 두 개의 셔플된 리스트가 구성됩니다.

        The index of each image in the images list will have the corresponding
        label at the same index in the labels list.
        이미지 목록의 각 이미지 인덱스는 레이블 목록의 동일한 인덱스에 해당 레이블을 갖습니다.
        """
        # csv 파일 읽기
        labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
        # label 정보가 담긴 txt 파일을 열어 각 글자를 원소로하는 리스트 생성
        labels_file = io.open(label_file, 'r',
                              encoding='utf-8').read().splitlines()

        # Map characters to indices.
        # 각 글자를 색인(index)와 맵핑하여 디렉토리 label_dict에 저장
        # ex) 가 - 0, 각 - 1, 간 - 2, ...
        label_dict = {}
        count = 0
        for label in labels_file:
            label_dict[label] = count
            count += 1

        # Build the lists.
        # 리스트를 생성
        images = []
        labels = []
        # csv 파일에서 각 행에서 이미지 경로와 label 색인 정보를 얻어 images와 labels에 각각 저장
        for row in labels_csv:
            file, label = row.strip().split(',')
            images.append(file)
            labels.append(label_dict[label])

        # Randomize the order of all the images/labels.
        # images와 labels의 순서를 섞음
        shuffled_indices = list(range(len(images)))
        random.seed(12121)
        random.shuffle(shuffled_indices)
        filenames = [images[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]

        # image 경로와 label 정보를 리턴
        return filenames, labels

    def write_tfrecords_file(self, output_path, indices):
        """
        Writes out TFRecords file.
        TFRecord 파일을 작성한다.
        """
        writer = tf.python_io.TFRecordWriter(output_path)
        for i in indices:
            filename = self.filenames[i]
            label = self.labels[i]
            with tf.gfile.FastGFile(filename, 'rb') as f:
                im_data = f.read()

            # Example is a data format that contains a key-value store, where
            # each key maps to a Feature message. In this case, each Example
            # contains two features. One will be a ByteList for the raw image
            # data and the other will be an Int64List containing the index of
            # the corresponding label in the labels list from the file.
            # Example은 키-값 저장을 포함하는 데이터 형식이며, 여기서 각 키는 Feature 메시지에 맵핑된다.
            # 이 경우 각 Example은 두 feature를 포함한다. 하나는 원시 이미지 데이터의 ByteList이고,
            # 다른 하나는 파일로부터 얻은 label list에 있는 해당 label의 인덱스를 포함하는 Int64List가 될 것이다.
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/class/label': _int64_feature(label),
                'image/encoded': _bytes_feature(tf.compat.as_bytes(im_data))}))
            writer.write(example.SerializeToString())
            self.counter += 1
            # 처리한 이미지 수 출력
            if not self.counter % 1000:
                print('Processed {} images...'.format(self.counter))
        writer.close()

    def convert(self):
        """This function will drive the conversion to TFRecords.  이 기능은 TFRecords로의 전환을 추진한다.

        Here, we partition the data into a training and testing set, then
        divide each data set into the specified number of TFRecords shards.
        여기서는 데이터를 학습용과 테스트용 셋으로 분할한 다음,
        각 데이터 셋을 지정된 수의 TFRecords 셰이드로 나눈다.
        """

        num_files_total = len(self.filenames)

        # Allocate about 15 percent of images to testing
        # 테스트를 위한 15퍼센트의 이미지가 실제로 몇 개인지 계산
        num_files_test = int(num_files_total * .15)

        # About 85 percent will be for training.
        # 트레이닝을 위한 85퍼센트의 이미지가 실제로 몇 개인지 계산
        num_files_train = num_files_total - num_files_test

        print('Processing training set TFRecords...')

        # shard 당 몇 개의 파일을 저장해야할 지 계산
        files_per_shard = int(math.ceil(num_files_train /
                                        self.num_shards_train))
        start = 0
        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir,
                                      'train-{}.tfrecords'.format(str(i)))
            # Get a subset of indices to get only a subset of images/labels for
            # the current shard file.
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_train, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'train-{}.tfrecords'.format(
                                            str(self.num_shards_train)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('Processing testing set TFRecords...')

        files_per_shard = math.ceil(num_files_test / self.num_shards_test)
        start = num_files_train
        for i in range(1, self.num_shards_test):
            shard_path = os.path.join(self.output_dir,
                                      'test-{}.tfrecords'.format(str(i)))
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_total, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'test-{}.tfrecords'.format(
                                            str(self.num_shards_test)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('\nProcessed {} total images...'.format(self.counter))
        print('Number of training examples: {}'.format(num_files_train))
        print('Number of testing examples: {}'.format(num_files_test))
        print('TFRecords files saved to {}'.format(self.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-label-csv', type=str, dest='labels_csv',
                        default=DEFAULT_LABEL_CSV,
                        help='File containing image paths and corresponding '
                             'labels.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecords files.')
    parser.add_argument('--num-shards-train', type=int,
                        dest='num_shards_train',
                        default=DEFAULT_NUM_SHARDS_TRAIN,
                        help='Number of shards to divide training set '
                             'TFRecords into.')
    parser.add_argument('--num-shards-test', type=int,
                        dest='num_shards_test',
                        default=DEFAULT_NUM_SHARDS_TEST,
                        help='Number of shards to divide testing set '
                             'TFRecords into.')
    args = parser.parse_args()
    converter = TFRecordsConverter(args.labels_csv,
                                   args.label_file,
                                   args.output_dir,
                                   args.num_shards_train,
                                   args.num_shards_test)
    converter.convert()
