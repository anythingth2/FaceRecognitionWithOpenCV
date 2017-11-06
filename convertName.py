import os

NEW_FACE_PATH = os.getcwd()+'\\datasets\\newfaces'

for folder in os.listdir(NEW_FACE_PATH):
    path = os.path.join(NEW_FACE_PATH,folder)
    img_files = os.listdir(path)

    i = 2
    for img_file_name in img_files:
        old_img_file_path = os.path.join(path,img_file_name)
        new_img_file_path = os.path.join(path,'subject'+folder +'.'+ str(i))
        print(old_img_file_path,new_img_file_path)
        os.rename( old_img_file_path, new_img_file_path )
        i+=1