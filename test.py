import pandas as pd
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import math
import random
from tqdm import tqdm
from mysql.connector import connect, Error
from getpass import getpass

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname,'data/')
os.chdir(filename)

HOST = 'localhost'
USER = 'root'
PASSWORD = getpass('Mot de passe : ')
DB_NAME = 'images'

get_all = """
SELECT path, red, green, blue
FROM image, medium_color
WHERE image.id = medium_color.image_id
"""

def chose_best_resolution(img, DESIRED_RESOLUTION_WIDTH, DESIRED_RESOLUTION_HEIGHT):
    img_width = img.width
    img_height = img.height
    return DESIRED_RESOLUTION_WIDTH, DESIRED_RESOLUTION_HEIGHT

def dominant_color(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    red, green, blue = colors[0][1][2], colors[0][1][1], colors[0][1][0]
    return red, green, blue
    
def img_k_means(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    cluster = KMeans(n_clusters=1).fit(reshape)
    return dominant_color(cluster, cluster.cluster_centers_)

def tile(im, i, j, max_iteration_height, max_iteration_width):
    left = j * im.width / max_iteration_width
    upper = i * im.height / max_iteration_height
    right = (j + 1) * im.width / max_iteration_width
    lower = (i + 1) * im.height / max_iteration_height
    return im.crop((left, upper, right, lower))

def get_closest_image (datas):
    return datas.loc[datas['d_point'] == min(datas['d_point'])].values[0]
    
def disance (expect_red, expect_green, expect_blue, test_red, test_green, test_blue):
    return math.sqrt((expect_red - test_red) ** 2 + (expect_green - test_green) ** 2 + (expect_blue - test_blue) ** 2)

def query_all_datas ():
    try:
        with connect(
            host=HOST,
            user=USER,
            password=PASSWORD,
            database=DB_NAME,
        ) as connection:
            return pd.read_sql(get_all, connection)               
    except Error as e:
        print(e)
    finally:
        connection.close()

rows = {
    'path': [],
    'red' : [],
    'green' : [],
    'blue' : []
}

print(' > Chosing best resolution to suit your desire...')
DESIRED_RESOLUTION_WIDTH, DESIRED_RESOLUTION_HEIGHT = 25,25
img_path = '11351119_782189315228604_7066093485773873391_n.jpg'
img = Image.open(img_path)
DESIRED_RESOLUTION_WIDTH, DESIRED_RESOLUTION_HEIGHT = chose_best_resolution(img, DESIRED_RESOLUTION_WIDTH, DESIRED_RESOLUTION_HEIGHT)

print(' > Fetching database...')
datas = query_all_datas()

print(' > Chosing right images...')
chosen_paths = []
for width in tqdm(range(DESIRED_RESOLUTION_WIDTH)):

    for height in range(DESIRED_RESOLUTION_HEIGHT):
        _img = tile(img, height, width, DESIRED_RESOLUTION_HEIGHT, DESIRED_RESOLUTION_WIDTH)
        test_blue, test_green, test_red = img_k_means(np.array(_img))
        datas['d_point'] = datas.apply(lambda x: disance(test_red, test_green, test_blue, x['red'], x['green'], x['blue']), axis = 1)
        chosen_paths.append(get_closest_image(datas)[0])

print(' > Assembling images...')
final_image = Image.new('RGB', img.size)
for width in tqdm(range(len(chosen_paths))):
    path = chosen_paths[width]
    img_to_add = Image.open(path)
    img_to_add = img_to_add.resize((int(img.width/DESIRED_RESOLUTION_WIDTH), int(img.height/DESIRED_RESOLUTION_HEIGHT)))
    height = int(width/)
    final_image.paste(img_to_add, (int(img.width/DESIRED_RESOLUTION_WIDTH) * height, int(img.height/DESIRED_RESOLUTION_HEIGHT) * width))

# print(' > Saving image...')
# final_image.save('result.jpg')
# print(' > Image saved at ', os.path.dirname(__file__) + 'result.jpg')