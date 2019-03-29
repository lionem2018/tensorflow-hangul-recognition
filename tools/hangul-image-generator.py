#!/usr/bin/env python

import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../image-data')

# Number of random distortion images to generate per font and character.
# 글꼴 및 문자 당 생성할 랜덤한 왜곡 이미지의 수
DISTORTION_COUNT = 3

# Width and height of the resulting image.
# 결과 이미지의 높이 너비
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.  한글 이미지 파일 생성

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    전달된 레이블 파일을 받아들여 글꼴 디렉토리에 제공된 글꼴 파일을 사용하여 여러 이미지를 생성합니다.
    글꼴 디렉토리는 * .ttf (True Type Font) 파일로 채워질 것으로 예상됩니다.
    생성된 이미지는 주어진 출력 디렉토리에 저장됩니다. 이미지 경로는 해당 레이블을 CSV 파일에 나열합니다.
    """

    # Label file에서 label 읽어들이기
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    # image 생성 디렉토리 지정 및 생성
    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    # ttf 파일 리스트 생성
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    # 각 ttf파일들을 통해 생성한 글자 이미지와 레이블 맵핑한 csv 파일 생성
    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
                         encoding='utf-8')

    total_count = 0
    prev_count = 0
    # label의 수 만큼 글자 생성
    for character in labels:
        # Print image count roughly every 5000 images.
        # 5000개 이상의 이미지를 생성할 때마다 현재 생성한 이미지의 수 출력
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        # 각 글자 이미지를 생성할 폰트 수 만큼 반복
        for font in fonts:
            total_count += 1
            # 흑백모드('L')의 64*64 이미지 생성
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            # 트루 타입의 폰트 파일(ttf)을 48 사이즈로 로드
            font = ImageFont.truetype(font, 48)
            # 주어진 이미지(image)를 그리는 객체 생성
            drawing = ImageDraw.Draw(image)
            # 주어진 string(여기서는 character)를 font 스타일로 그렸을 때의 크기(높이, 낮이)를 리턴
            w, h = drawing.textsize(character, font=font)
            # 그릴 글자의 상단 왼쪽 코너의 위치, 그릴 글자, 글자를 채울 색, 폰트 스타일을 지정하여 그림
            # image에 생성
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                character,
                fill=(255),
                font=font
            )
            # 각 폰트의 글자 이미지 file 이름 생성
            file_string = 'hangul_{}.jpeg'.format(total_count)
            # 글자 이미지 path 지정
            file_path = os.path.join(image_dir, file_string)
            # JPEG 형식으로 글자 이미지 생성
            image.save(file_path, 'JPEG')
            # csv 파일에 생성한 글자 이미지와 해당 글자(label)를 나란히 파일에 작성
            labels_csv.write(u'{},{}\n'.format(file_path, character))

            # 실제 폰트와는 조금 다른, 왜곡된 글자 이미지 생성
            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'hangul_{}.jpeg'.format(total_count)
                file_path = os.path.join(image_dir, file_string)
                # 원래 스타일의 글자 데이터를 array 형태로 복사
                arr = numpy.array(image)

                # array 형으로 이미지 왜곡 수행
                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                # array 형의 데이터를 image로 전환
                distorted_image = Image.fromarray(distorted_array)
                # 왜곡 이미지 저장
                distorted_image.save(file_path, 'JPEG')\
                # csv 파일에 생성한 글자 이미지와 해당 글자(label)를 나란히 파일에 작성
                labels_csv.write(u'{},{}\n'.format(file_path, character))

    print('Finished generating {} images.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.  이미지에 elastic 왜곡 수행

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    여기서 alpha는 변형의 강도를 제어하는 배율 인수를 나타냅니다.
    sigma는 가우스 필터 표준 편차를 나타냅니다.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)
