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
    # res6 = cursor.query("SELECT * FROM Image").df()
    # print(res6)

    # List all the built-in functions in EvaDB
    # print(cursor.query("SHOW FUNCTIONS;").df())

    cursor.query("DROP FUNCTION IF EXISTS greyImage;").df()
    cursor.query(f"CREATE FUNCTION greyImage IMPL  './greyImage.py';").df()

    cursor.query("DROP FUNCTION IF EXISTS invertImage;").df()
    cursor.query(f"CREATE FUNCTION invertImage IMPL  './invertImage.py';").df()

    kernel = 25
    border = 0
    cursor.query("DROP FUNCTION IF EXISTS blurImage;").df()
    cursor.query(f"""
                    CREATE FUNCTION blurImage 
                    IMPL  './blurImage.py' 
                    kernel '{kernel}' 
                    bordertype '{border}';""").df()

    cursor.query("DROP FUNCTION IF EXISTS invertblurImage;").df()
    cursor.query(f"CREATE FUNCTION invertblurImage IMPL  './invertBlur.py';").df()

    prompts = user_input()

    if prompts[0] == 'Y':
        res4 = cursor.query(""" 
                            SELECT img.data, greyImage(img.data), invertblurImage(blur.data) FROM Image as img
                            JOIN LATERAL greyImage(img.data) AS grey(data)
                            JOIN LATERAL invertImage(grey.data) AS invert(data)
                            JOIN LATERAL blurImage(invert.data) AS blur(data)
                            """).df()

        img = cv2.divide(res4["greyimage.greyframe"].iloc[0], res4["invertblurimage.invertblurframe"].iloc[0], scale=256.0)
        greyimg = res4["greyimage.greyframe"].iloc[0]

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if greyimg[i][j] < 70 and img[i][j] > 200:
                    img[i][j] = 20
                if img[i][j] < 230:
                    img[i][j] = 30
        
        # fileName = 'Images\\smilingteen.jpeg'
        # cursor.query(f"INSERT INTO Image (name, data) VALUES ('name', '{img}')").df()

        cv2.imwrite('./tempImage/temp.jpeg', img)
        cursor.query("DROP TABLE IF EXISTS Image;").df()
        cursor.query("LOAD IMAGE 'tempImage/*.jpeg' INTO Image").df()

    if prompts[1] != None:
        cursor.query("DROP FUNCTION IF EXISTS controlNet;").df()
        cursor.query(f"""
                     CREATE FUNCTION controlNet
                     IMPL  './ControlNet.py'
                     prompt '{prompts[1]}'
                     a_prompt '{prompts[2]}'
                     n_prompt '{prompts[3]}'
                     ddim_steps '{prompts[4]}';
                     """).df()

        res5 = cursor.query("SELECT controlNet(img.data) FROM Image as img ").df()
        img = res5["controlnet.controlframe"].iloc[0]
    
    # Show the image
    fig, ax = plt.subplots(1, figsize=(15, 15))

    ax.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def user_input():
    prompts = []
    print("Welcome! Please answer the following questions to alter the image as you desire.")
    print('Would you like to create a sketch of the image?')

    while True:
        while True:
            answer = input('Y/N: ').strip()
            if answer == 'Y' or answer == 'N':
                prompts.append(answer)
                break
            else:
                print("Invalid input")
        
        print("Enter a prompt to alter the image by.")
        print("If you don't want a prompt, type 'N'")
        while True:
            answer = input('Prompt: ').strip()
            if answer == 'N':
                prompts.append(None)
            else:
                prompts.append(answer)
            break
        if prompts[0] == 'N' and prompts[1] == None:
            print("The image needs to be altered. Please select sketch, prompt, or both.")
            prompts = []
        else:
            prompts.append("best quality, extremely detailed")
            prompts.append("longbody, lowres, " \
              "bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
            prompts.append('20')
            if prompts[0] == 'Y' and prompts[1] != None:
                prompts[1] = "Pencil Sketch, High Contrast, detailed line art, fine details, " + prompts[1]
                prompts[2] = "Black Lines, Plenty White, Minimalistic, Empty, Defined Lines, High Contrast, " + prompts[2]
                prompts[3] = "Color, Blurry, Gray, Light Gray, Dark Gray, " + prompts[3]
                prompts[4] = '16'
            print("Thank you! Your image will now be processed.")
            return prompts

if __name__ == "__main__":
    main()