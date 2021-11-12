from mysql.connector import connect, Error
from getpass import getpass
import os
import numpy as np
from skimage import io
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm

HOST = 'localhost'
USER = 'root'
PASSWORD = getpass('Mot de passe : ')
DB_NAME = 'images'

create_image_table = """
CREATE TABLE image(
    id INT AUTO_INCREMENT PRIMARY KEY,
    path VARCHAR(100)
)
"""

create_medium_color_table = """
CREATE TABLE medium_color (
    image_id INT,
    red INT,
    green INT,
    blue INT,
    FOREIGN KEY(image_id) REFERENCES image(id),
    PRIMARY KEY(image_id)
)
"""

insert_image_query = """
INSERT INTO image
(path)
VALUES ( %s )
"""
get_image_query = """
SELECT id, path
FROM image
WHERE path IS NOT NULL
"""

insert_medium_color_query = """
INSERT INTO medium_color
(red, green, blue, image_id)
VALUES ( %s, %s, %s, %s )
"""

# Add all path  from .\data\*.png .\data\*.jpg to image table
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname,'data/')
image_data = [tuple(s for s in i.split(',')) for i in os.listdir(filename)]

def dominant_color(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    red, green, blue = colors[0][1][2], colors[0][1][1], colors[0][1][0]
    return red, green, blue

def compute_medium_value():
    try:
        with connect(
            host=HOST,
            user=USER,
            password=PASSWORD,
            database=DB_NAME,
        ) as connection:
            with connection.cursor() as cursor:
                os.chdir('data/')
                cursor.execute(get_image_query)
                paths = cursor.fetchall()
                colors_data = []
                print(' > Database fills ...')
                for path in tqdm(paths):
                    current_id = path[0]
                    image = cv2.imread(path[1])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
                    cluster = KMeans(n_clusters=1).fit(reshape)
                    red, green, blue = dominant_color(cluster, cluster.cluster_centers_)
                    colors_data.append(tuple([red, green, blue, current_id]))
                print(' > Saving Database ...')
                cursor.executemany(insert_medium_color_query, colors_data)
                connection.commit()
    except Error as e:
        print(e)

print(' > Create database structure...')
try:
    with connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DB_NAME,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_image_table)
            cursor.execute(create_medium_color_table)
            cursor.executemany(insert_image_query,image_data)
            connection.commit()
except Error as e:
    print(e)

compute_medium_value()

print(' > Successfully ended!!!')