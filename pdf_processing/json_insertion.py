from google.oauth2 import service_account
from googleapiclient.discovery import build


import json
from termcolor import colored

'''
STEPS TO SET UP GOOGLE CLOUD API (for google drive and sheets)

1. Go to Google Cloud Console: https://console.cloud.google.com
2. Set up account according to instructions from Google
3. Create a new project
4. After setting up the project, go back to console main page, and click on "IAM & Admin"
    a. click on Service Accounts
    b. Create service account, dont have to fill out Permisions and Principles with access
    c. After the service account is created, click on the three dots below Action, and click on Manage Keys
    d. Click Add key, and Create new key, chose Key type to be JSON and create
    e. Download the JSON file, this is your service account file
5. Go back to the main page of the console, and in the search bar look up Google Sheets API and Google Drive API. 
    Click on the first result for both searches, and click Enable API

'''



# to use must get service account from google cloud console 
SERVICE_ACCOUNT_FILE = 'pdf_processing/service-account.json' #replace with your own to work
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]

# Auth
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
serviceDrive = build('drive', 'v3', credentials=creds)


# accessing the folder with the json files in google drive
folder_id = '1rWcAg2tH6U3YUVwxDp8IOWCN0dt3OGIo'

results = serviceDrive.files().list(
    q=f"'{folder_id}' in parents and trashed = false",
    spaces='drive',
    fields="files(id, name)",
    pageSize=20
).execute()

items = results.get('files', [])

# key: file name, minus '.json' and 'cleaned_'
# value:['link of cleaned json file', 'link of regular json file']
title_links = {} 

for item in items:
    file_name = item['name']
    file_link = item['id']
    title = file_name[:-5] #taking off the ".json" at end

    # cleaned version
    if file_name[:8] == "cleaned_": 
        title = title[8:]
        if title not in title_links:
            title_links[title] = [file_link, '']
        else:
            title_links[title][0] = file_link

    # regular version
    else:
        if title not in title_links:
            title_links[title] = ['', file_link]
        else:
            title_links[title][1] = file_link


   
matches = {}
with open('pdf_processing/matches.json', 'r') as data:
    matches = json.load(data)



# accessing the agroforestry data spreadsheet using sheets API
SPREADSHEET_ID = '1aKZ_xFZNCEuihLo2dXzU07KwL5sDyAbD2Fod4FmE6FE' # This spreadsheet must be shared with the service account email
# ^Note that this spreadsheet is a copy of the original because I didn't want to mess that one up

serviceSheets = build('sheets', 'v4', credentials=creds)

manual_insertion = {}

# inserting the google drive link of the json file into spreadsheet
for name in title_links:
    clean = title_links[name][0]
    regular = title_links[name][1]

    clean_link = f"https://drive.google.com/file/d/{clean}/view" #generating link for the json 
    regular_link= f"https://drive.google.com/file/d/{regular}/view"
    values =[[clean_link, regular_link]]
    
    #catching the titles that are not able to be matched, will manually insert later
    if name not in matches:
        print(colored("Unable to find", "red"), name)
        manual_insertion[name] = [clean_link, regular_link]

    else:    
        row = matches[name]["row"]
        print("Inserting into row", row)

        # specify the cells to insert into
        updated_cell = f"S1 Literature!M{row}:N{row}" #I just chose the two blank columns at the end of the spreadsheet
        body = {
            'values': values
        }

        #actually inserting the value
        result = serviceSheets.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=updated_cell,
            valueInputOption= 'RAW',  
            body=body
        ).execute()

