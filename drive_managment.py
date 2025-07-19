from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def authorize_gdrive():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

def empty_gdrive_trash(drive):
    trash_list = drive.ListFile({'q': "trashed=true"}).GetList()

    for file in trash_list:
        try:
            file.Delete()
        except Exception as e:
            print(f"Ошибка при удалении {file['title']}: {e}")
