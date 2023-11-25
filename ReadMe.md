# EvaDB Project 2: ControlNet with Artline
## Introduction:
This project takes in a profile image of a person and can convert the style to a new image using ControlNet. EvaDB is utilized to perform CV2 functions through queries on an image for ArtLine sketches as well as hosting ControlNet in a separate function. 

Currently, the application runs by taking in an image from the Image folder. It will then ask the user if it wants to run a lineart sketch or ControlNet prompt. 
If a sketch is used, the following occurs to the image: Turns the image into grayscale, inverts the grayscale image, Gaussian Blurs the inverted grayscale image, then inverts the result. The grayscaled image is then divided by the grayscaled-inverted-blurred-inverted image. After this, this new image is compared with the original image data and grayscale data to fill in empty spaces left by the above functionality. This produces the final image, which an example can be seen below.

## Example Output:
![alt text](https://github.com/JohnK70/EvaDBP1Artline/blob/main/githubImage.png?raw=true)

If the ControlNet prompt is used, the app will query the ControlNet function. It will then process the image through Stable Diffusion and the Canny control method. This produces the final image, an example of which can be seen below.

## Example Output:
Prompt: Beach Background, Young Boy
![alt text](https://github.com/JohnK70/EvaDB-ControlNet-Artline/blob/main/ControlNetTeen.png?raw=true)

Prompt: City background, Bold Colors, Defined Woman
![alt text](https://github.com/JohnK70/EvaDB-ControlNet-Artline/blob/main/modelCityWoman.png?raw=true)

ControlNet can also be used with the original ArtLine functionality together. An example of an output is seen below.

## Example Output:
Prompt: Pencil Sketch of Man
![alt text](https://github.com/JohnK70/EvaDB-ControlNet-Artline/blob/main/DrakeControlArtLine.png?raw=true)

### Running the app:
Ensure you have Python 3.10. This is required for all packages to run properly.

Clone the github repo into your desired location. Then use <pip install -r requirements.txt> inside the directory to install requirements.

---------------------------------------------------------------------------------------------------------------------------------------
Two models will need to be installed and put in the models folder. These are v1-5-pruned.ckpt and control_v11p_sd15_canny.pth. The download links can be found for them below:

v1-5-pruned.ckpt: https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt?download=true

control_v11p_sd15_canny.pth: https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth?download=true

---------------------------------------------------------------------------------------------------------------------------------------

There are three image based folders related to the repository: Images, TestImages, and TempImage.
TestImages holds all images that you want to use later. Use this for storing any images. It will not be used when running the code.
TempImage is used if both ArtLine and ControlNet are used at the same time. **Do not alter this folder.**
Images will hold the single image you want to convert using LineArt or ControlNet. **Only put one image in this folder.**


To use the app, make sure the Images folder is clear except for the one image you want to alter.
After this, you can run the app by running the file EvaDBArtline.py
The app will prompt you to use either LineArt, ControlNet, or both. Follow the prompts in the terminal to select the options.
MatPlotLib will then display the outputted final image. This will be deleted once you close out of the image, so if you want to save it you must do so manually.
