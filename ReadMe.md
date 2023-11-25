# EvaDB Project 2: ControlNet with Artline
## Introduction:
This project takes in a profile image of a person and can convert the style to a new image using ControlNet. EvaDB is utilized to perform CV2 functions through queries on an image for ArtLine sketches as well as hosting ControlNet in a separate function. 

Currently, the application runs by taking in an image from the Image folder. It will then ask the user if it wants to run a lineart sketch or ControlNet prompt. 
If a sketch is used, the following occurs to the image: Turns the image into grayscale, inverts the grayscale image, Gaussian Blurs the inverted grayscale image, then inverts the result. The grayscaled image is then divided by the grayscaled-inverted-blurred-inverted image. After this, this new image is compared with the original image data and grayscale data to fill in empty spaces left by the above functionality. This produces the final image, which an example can be seen below.

## Example Output:
![alt text](https://github.com/JohnK70/EvaDBP1Artline/blob/main/githubImage.png?raw=true)

If the ControlNet prompt is used, then the app will then query the ControlNet function. It will then process the image through Stable Diffusion and the Canny control method. This produces the final image, an example of which can be seen below.

## Example Output:

### Running the app:
Ensure you have Python 3.10. This is required for all packages to run properly.

Clone the github repo into your desired location. Then use <pip install -r requirements.txt> inside the directory to install requirements.

There are two folders related to the repository: Images and TestImages.
TestImages holds all images that you want to use later. Use this for storing any images. It will not be used when running the code.
Images will hold the single image you want to convert into lineart.

To use the app, make sure the Images folder is clear except for the one image you want to turn into lineart.
After this, you can run the app by running the file EvaDBArtline.py
MatPlotLib will then display the outputted lineart image. This will be deleted once you close out of the image, so if you want to save it you must do so manually.

### Future Work:
* Fine tune model to produce outputs closer to ArtLine
* Add ControlNet functionality (prompts which change the resulting image)
