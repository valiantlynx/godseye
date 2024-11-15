from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import base64
from email.message import EmailMessage
import os

SCOPES = ['https://mail.google.com/']  # Or use ['https://www.googleapis.com/auth/gmail.send']
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_FILE = os.path.join(MODULE_DIR, 'client_secret.json')
TOKEN_FILE = os.path.join(MODULE_DIR, 'token.json')


def send_gmail(recipient, message_text = "Violent behaviour detected!"):
    creds = None
    if not recipient:
        return None

    # Check if token.json exists and load credentials from it
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid credentials are available, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()
        message.set_content(message_text)

        message["To"] = recipient
        message["From"] = "godseye@gmail.com"
        message["Subject"] = "Violence detected"

        # Encode the message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}

        # Send the message
        send_message = service.users().messages().send(userId="me", body=create_message).execute()
        print(f'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(f"An error occurred: {error}")
        send_message = None
    return send_message

if __name__ == "__main__":
    send_gmail()
