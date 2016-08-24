#!/usr/bin/python

import sys
import os
import datetime
import smtplib
from email.mime.text import MIMEText

EMAIL_LIST = []
EMAIL_LIST.append("lukaszad@gmail.com")
EMAIL_LIST.append("johannes.eichstaedt@gmail.com")
EMAIL_LIST.append("andy.schwartz@gmail.com")

def sendEmail(subject, message, email_address):
    message = quote(message)
    subject = quote(subject)
    cmd = "echo "+message+" | mailx -s "+subject+" " + email_address
    print(( "command issued, [["+cmd+"]]"))
    os.system(cmd)

def sendEmail(subject, message, email_address):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = 'featureWorker.py'
    msg['To'] = email_address
    s = smtplib.SMTP('localhost')
    s.sendmail('featureWorker.py', email_address, msg.as_string())
    s.quit()


def sendEmails(subject, message, email_addresses):
    for email_address in email_addresses:
        sendEmail(subject, message, email_address)
