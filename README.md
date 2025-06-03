# AI Black & White to Color Image Restoration

This project uses a convolutional neural network (CNN) to automatically restore and colorize black & white images. The web application is built using Flask, allowing users to upload grayscale images and view the colorized results interactively.

# Repository Structure
- images  
  Contains sample black & white images used for testing and demonstration purposes.

- templates  
  Contains HTML files and related assets for the Flask web application, including the result page that displays the colorized images.

- final_code.py  
  The main Python script that runs the Flask web app and loads the pre-trained colorization model.

- model 
  Contains the model files required for image colorization: (large file- use git lfs)

- static
  contains the background img for webpage.


# About Large Files in this Project
This project uses (Git LFS) to handle large model files:
- `model/colorization_release_v2.caffemodel`
Git LFS stores pointers in the repository instead of the full files to keep the repo size manageable.

If you're cloning this repo for the first time, make sure to have [Git LFS installed](https://git-lfs.github.com/) so the large files download correctly.

# How to install Git LFS
On Windows Git Bash or terminal
git lfs install

- After cloning, Git LFS will automatically download the large model files. If you face any issues, you can run this on terminal:  
  git lfs install
  git lfs pull

# Run the model 
You can simply install the py dependecies and run it via VScode, it will open in localhost: http://127.0.0.1:5000

or 

You can run it locally from your terminal using these commands:

cd "location"           
python -m venv venv          
venv\Scripts\activate       

pip install -r requirements.txt    (If you don't have a requirements.txt, then install packages manually)

pip install flask numpy opencv-python matplotlib

python final_code.py         
