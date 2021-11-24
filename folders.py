import local_variables
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split



df = pd.read_csv("labels_Pictures_LE.csv")
list_no_v =[]
list_one_v = []
list_two_v = []



def createList():
    for i in range(0,len(df.index)):
        name = df.loc[i]["bee_id"]
        cate = df.loc[i]["class"]
        if int(cate) == 0:
            list_no_v.append(name)
        elif int(cate) == 1:
            list_one_v.append(name)
        else:
            list_two_v.append(name)
    print("Lists done!")


def create_folders():
    pth = local_variables.pictures_le_filepath
    os.mkdir(pth + "_train\\")
    os.mkdir(pth + "_test\\")
    pth2 = pth + "_train\\"
    os.mkdir(pth2 + "no\\")
    os.mkdir(pth2 + "one\\")
    os.mkdir(pth2 + "two\\")
    pth2 = pth + "_test\\"
    os.mkdir(pth2 + "no\\")
    os.mkdir(pth2 + "one\\")
    os.mkdir(pth2 + "two\\")
    print("Folders done!")

def create_folder():
    Notrain, Notest = train_test_split(list_no_v, test_size=0.20, shuffle=False)
    Onetrain, Onetest = train_test_split(list_one_v, test_size=0.20, shuffle=False)
    Twotrain, Twotest = train_test_split(list_two_v, test_size=0.20, shuffle=False)



    for subdir, dirs, files in os.walk(local_variables.pictures_le_filepath):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filename in list_no_v:
                if filename in Notrain:
                    print("filepath: " + filepath)
                    newfilepath = subdir + '_train' + os.sep + 'no' + os.sep
                    print("newfilepath: " + newfilepath)
                    print("filename: " + filename)
                    shutil.copy(filepath,newfilepath)
                else:
                    newfilepath = subdir +'_test' + os.sep + 'no' + os.sep
                    shutil.copy(filepath,newfilepath)
            elif filename in list_one_v:
                if filename in Onetrain:

                    newfilepath = subdir + '_train' + os.sep + 'one' + os.sep
                    shutil.copy(filepath,newfilepath)
                else:
                    newfilepath = subdir + '_test' + os.sep + 'one' + os.sep
                    shutil.copy(filepath,newfilepath)
            else:
                if filename in Twotrain:
                    newfilepath = subdir + '_train' + os.sep + 'two' + os.sep
                    shutil.copy(filepath,newfilepath)
                else:
                    newfilepath = subdir + '_test' + os.sep + 'two' + os.sep
                    shutil.copy(filepath,newfilepath)
        print("test/train folders done!")