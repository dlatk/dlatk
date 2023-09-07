import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
from google.colab import auth, drive, files

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
  print("File %s uploaded successfully.\n File ID is %s - store this somewhere to fetch your file later." % (file["title"], file["id"]))

def get_file_from_drive(filename):

  #1. Authenticate to Google Drive
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  #2. Auto-iterate through all files in the root folder and get the fileid
  files = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
  fileid = [file["id"] for file in files if file["title"] == filename][0]

  #2. Download the dataset from Google Drive to Colab
  downloaded = drive.CreateFile({"id": fileid})
  downloaded.GetContentFile(filename)
