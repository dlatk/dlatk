import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from google.colab import auth, drive, files
from IPython import display, get_ipython
from IPython.display import Javascript

def upload_dataset():

  #1. Authenticate to Google Drive
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  #2. Upload the dataset to Google Colab - /content/
  file = files.upload()
  filename = list(file.keys())[0]
  print("Uploading ", filename)

  #3. Copy the local file to Google Drive for later use
  file = drive.CreateFile({'title': filename})
  file.SetContentFile(os.path.join("/content", filename))
  file.Upload()
  print("File %s uploaded successfully." % (file["title"]))

def get_file_from_drive(filename=None):

  #1. Authenticate to Google Drive
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  #2. Get fileid either for the custom dataset or the tutorial dataset
  if filename is None:

    TUTORIAL_FOLDER = "colab_csv"
    TUTORIAL_DATASET = "msgs100"

    query = "(title='{}') and (mimeType='application/vnd.google-apps.folder') and (trashed=false) and (sharedWithMe=True)"
    folderid = drive.ListFile({'q': query.format(TUTORIAL_FOLDER)}).GetList()[0]["id"]
    fileid = drive.ListFile({'q': "(title contains '{}') and ('{}' in parents) and (trashed=false)".format(TUTORIAL_DATASET, folderid)}).GetList()[0]["id"]

  else:
    #Auto-iterate through all files in the root folder and get the fileid
    files = drive.ListFile({'q': "(title contains '{}') and ('root' in parents) and (trashed=false)".format(filename.split('.')[0])}).GetList()
    fileid = [file["id"] for file in files if ".csv" in file["title"]][0]

  #3. Download the dataset from Google Drive to Colab
  downloaded = drive.CreateFile({"id": fileid})
  downloaded.GetContentFile(filename)
  print("File %s copied successfully from Google Drive." % (file["title"]))

def resize_colab_output():

    #https://github.com/googlecolab/colabtools/issues/541
    def resize_output():
        display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 400})'''))

    get_ipython().events.register("pre_run_cell", resize_output)
