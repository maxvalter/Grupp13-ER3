from PIL import Image,ImageEnhance
import os
import numpy
import local_variables

PATH = local_variables.pictures_le_filepath + "_edits"
DIR = PATH + os.sep

def edit_image(filename,filepath):
    im = Image.open(filepath)
    im.resize((64,128))
    image_data = im.load()
    height,width = im.size
    for loop1 in range(height):
        for loop2 in range(width):
            r,g,b = image_data[loop1,loop2]
            image_data[loop1,loop2] = r,int(g*1.5),int(b*1.5)
    im.save(DIR + filename + ".png")


def make_edit_folder():
    os.mkdir(PATH)

def loop_folder(folderpath):
    for filename in os.listdir(local_variables.pictures_le_filepath):
        edit_image(filename,local_variables.pictures_le_filepath + os.sep + filename)
    print("Editing images done!")
