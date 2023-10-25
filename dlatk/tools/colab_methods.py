import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.axes import Axes

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from google.colab import auth, drive, files
from IPython.core.display import display
from IPython import get_ipython
from IPython.display import Javascript

def upload_dataset(filename=None):

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
    #Auto-iterate through all files in the root folder and get the fileid
    file_list = drive.ListFile({'q': "(title contains '{}') and ('root' in parents) and (trashed=false)".format(filename.split('.')[0])}).GetList()
    fileid = [file["id"] for file in file_list if ".csv" in file["title"]][0]

    #Download the dataset from Google Drive to Colab
    downloaded = drive.CreateFile({"id": fileid})
    downloaded.GetContentFile(filename)
    print("File %s copied successfully from Google Drive." % (filename))

def shorten_colab_output():

    #https://github.com/googlecolab/colabtools/issues/541
    def resize_output():
        display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 250})'''))

    get_ipython().events.register("pre_run_cell", resize_output)

def print_wordclouds(wordcloud_folder):

    images = glob.glob(os.path.join(wordcloud_folder, "*.png"))
    if len(images) == 0:
      print("None of the features were significant, hence wordclouds not produced.")
      return
    
    outcomes = set([re.split(r'_pos|_neg', image.split('/')[-1])[0] for image in images])
    outcome_image_map = {outcome: [image for image in images if (outcome + '_') in image.split('/')[-1]] for outcome in outcomes}

    for outcome in outcomes:

      outcome_images = outcome_image_map[outcome]
      fig, axes = plt.subplots(1, len(outcome_images), figsize=(5 * len(outcome_images), 5)) 
      axes = [axes] if isinstance(axes, Axes) else axes

      for i, image in enumerate(outcome_images):
        axes[i].set_axis_off()
        axes[i].set_title(image.split('/')[-1])
        axes[i].imshow(mpimg.imread(image))

    return
