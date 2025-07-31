import glob
import os
import re
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.axes import Axes

import urllib.error
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from google.colab import auth, drive, files
from IPython.core.display import display
from IPython import get_ipython
from IPython.display import Javascript, HTML

def upload_dataset(filename=None, foldername=None):

  #Authenticate to Google Drive
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  if filename is None:
    #Upload the dataset to Google Colab - /content/
    file = files.upload()
    filename = list(file.keys())[0]
    print("Uploading ", filename)

    #Copy the local file to Google Drive for later use
    file = drive.CreateFile({'title': filename})
    file.SetContentFile(os.path.join("/content", filename))
    file.Upload()
    print("File %s uploaded successfully." % (file["title"]))

  else:

    def search(name, parent='root'):

      query = "'{}' in parents and trashed=false".format(parent)
      try:

        found = drive.ListFile({'q': query}).GetList()
        folder = None
        for file_obj in found:
          if file_obj['mimeType'] == 'application/vnd.google-apps.folder':
            folder = file_obj if file_obj["title"] == name else search(name, file_obj['id'])
            if folder: return folder

        return folder

      except urllib.error.HttpError as e:
        return None

    # Recursively find the folder
    if foldername is not None:
      folder = search(foldername)
      if folder is None:
        print("Folder not found")
        return

      folder_id = folder["id"]

    else:
      folder_id = "root"

    # Auto-iterate through all files in the folder and get the fileid
    file_list = drive.ListFile({'q': "(title contains '{}') and ('{}' in parents) and (trashed=false)".format(filename, folder_id)}).GetList()
    if not file_list:
      print("File not found")
      return

    # Finally download the file if found
    file_id = [file["id"] for file in file_list if ".csv" in file["title"]][0]
    downloaded = drive.CreateFile({"id": file_id})
    downloaded.GetContentFile(filename)
    print("File %s copied successfully from Google Drive." % (filename))

def shorten_colab_output(pixels):

    #https://github.com/googlecolab/colabtools/issues/541
    def resize_output():
        display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: %d})''' % pixels))

    get_ipython().events.register("pre_run_cell", resize_output)

def change_bg_color(r, g, b):

  def change_color():
    display(HTML('''<style> body {background-color: rgb(%d, %d, %d);}</style>''' % (r, g, b)))

  get_ipython().events.register("pre_run_cell", change_color)

def colab_shorten_and_bg(pixels=250, r=255, g=250, b=232):
    shorten_colab_output(pixels)
    change_bg_color(r, g, b)

def print_wordclouds(wordcloud_folder):

    images = glob.glob(os.path.join(wordcloud_folder, "*.png"))
    if len(images) == 0:
      print("None of the features were significant, hence wordclouds not produced.")
      return
    
    outcomes = set([re.split(r'_pos|_neg', image.split('/')[-1])[0] for image in images])
    outcome_image_map = {outcome: [image for image in images if (outcome + '_') in image.split('/')[-1]] for outcome in outcomes}

    for outcome in outcomes:

      outcome_images = outcome_image_map[outcome]
      fig, axes = plt.subplots(1, len(outcome_images), figsize=(3.5 * len(outcome_images), 3)) 
      axes = [axes] if isinstance(axes, Axes) else axes

      outcome_images = sorted(outcome_images)
      for i, image in enumerate(outcome_images):
        axes[i].set_axis_off()
        axes[i].set_title(image.split('/')[-1])
        axes[i].imshow(mpimg.imread(image))

    return

def print_topic_wordclouds(wordcloud_folder, num_topics, images_per_row):

    images = glob.glob(os.path.join(wordcloud_folder, '*.png'))
    to_display = images[:min(num_topics, len(images))]

    fig, axes = plt.subplots(math.ceil(num_topics/images_per_row), images_per_row, figsize=(18, 8))
    for index, image in enumerate(images):

        axes[int(index/images_per_row), index%images_per_row].set_axis_off()
        axes[int(index/images_per_row), index%images_per_row].set_title(image.split('/')[-1])
        axes[int(index/images_per_row), index%images_per_row].imshow(mpimg.imread(image))

    return
