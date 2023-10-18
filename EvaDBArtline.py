import sys
import evadb
import matplotlib.pyplot as plt
import os
import shutil
import cv2
import numpy as np

def main():
    # Connect to EvaDB and get a database cursor for running queries
    cursor = evadb.connect().cursor()

    cursor.query("DROP TABLE IF EXISTS Image;").df()
    cursor.query("LOAD IMAGE 'Images/*.jpeg' INTO Image").df()

    # List all the built-in functions in EvaDB
    # print(cursor.query("SHOW FUNCTIONS;").df())

    cursor.query("DROP FUNCTION IF EXISTS greyImage;").df()
    cursor.query(f"CREATE FUNCTION greyImage IMPL  './greyImage.py';").df()

    cursor.query("DROP FUNCTION IF EXISTS invertImage;").df()
    cursor.query(f"CREATE FUNCTION invertImage IMPL  './invertImage.py';").df()

    cursor.query("DROP FUNCTION IF EXISTS blurImage;").df()
    cursor.query(f"CREATE FUNCTION blurImage IMPL  './blurImage.py';").df()

    cursor.query("DROP FUNCTION IF EXISTS invertblurImage;").df()
    cursor.query(f"CREATE FUNCTION invertblurImage IMPL  './invertBlur.py';").df()

    # res4 = cursor.query("SELECT img.data FROM Image as img JOIN LATERAL greyImage(img.data) as grey").df()
    res4 = cursor.query(""" 
                        SELECT img.data, greyImage(img.data), invertblurImage(blur.data) FROM Image as img
                        JOIN LATERAL greyImage(img.data) AS grey(data)
                        JOIN LATERAL invertImage(grey.data) AS invert(data)
                        JOIN LATERAL blurImage(invert.data) AS blur(data)
                        """).df()
    
    # print(res4["greyimage.greyframe"])
    img = cv2.divide(res4["greyimage.greyframe"].iloc[0], res4["invertblurimage.invertblurframe"].iloc[0], scale=256.0)
    greyimg = res4["greyimage.greyframe"].iloc[0]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if greyimg[i][j] < 70 and img[i][j] > 100:
                img[i][j] = 20
            if img[i][j] < 100:
                img[i][j] = 30



    # Show the image
    fig, ax = plt.subplots(1, figsize=(15, 15))

    ax.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # ax.imshow(res4["invertblurimage.invertblurframe"].iloc[0], cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    

if __name__ == "__main__":
    main()