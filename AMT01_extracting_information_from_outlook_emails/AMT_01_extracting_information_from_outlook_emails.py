# https://youtu.be/50o6RTvYIpY
"""
Automating manual tasks - DigitalSreeni
Reading Outlook inbox (or other mail folder) and extracting 
the required information. 

"""

import win32com.client
import pandas as pd #To capture data into a DataFrame

#Define the outlook object 
outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

#Print Outlook Folders
# for folder in outlook.Folders: 
#     print(folder.Name)

#Define the folder you want to search (inbox default is 6)
inbox = outlook.GetDefaultFolder(6) 

#Fetch messages from the above defined folder                                    
messages = inbox.Items

"""
#My test emails look like this...

Subject: From Sreeni

Name: John Smith
Company: ABC international
Email: test1@test.com
Message: Hi there, getting in touch


I sent a few emails in this format so we can extract various details and 
capture them into an Excel document. 

"""

#Let us create empty lists to capture the data iteratively
#NOTE that there are many ways to doing the following task.
all_names=[]
all_company = []
all_email = []
all_body_text = []

#Iteratively go through our messages to extract desired information
#Here, I am looking for emails with subject "From Sreeni"
for i, message in enumerate(messages):
    
    if message.Subject == "From Sreeni":     #Fetch all emails with this subject
        #print(message.SenderName)
        #date_time = message.LastModificationTime  #Extract the date & time the email was received
        
        text = message.Body  #Read the body of our email. 
        
        #Now extract the information we need from the message body
        ##Find the text "Name:" and read until it sees text "Company"
        #Repeat the same for other details
        name = text[text.rfind('Name:')+6:text.rfind('Company:')]  
        company = text[text.rfind('Company:')+9:text.rfind('Email:')]
        email = text[text.rfind('Email:')+7:text.rfind('Message:')]
        body_text = text[text.rfind('Message:')+9:]
        
        #Append identified text into our empty lists for each task. 
        all_names.append(name)
        all_company.append(company)
        all_email.append(email)
        all_body_text.append(body_text)

#Capture all information into pandas dataframe
extracted_info = pd.DataFrame(columns=["Name", "Company", "Email", "Message"])

extracted_info["Name"] = all_names
extracted_info["Company"] = all_company
extracted_info["Email"] = all_email
extracted_info["Message"] = all_body_text

#Save the dataframe as an excel document. 
extracted_info.to_excel("extracted_info.xlsx")
