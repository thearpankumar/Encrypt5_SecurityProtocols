import os
from dotenv import load_dotenv
load_dotenv()

# Application (client) ID of app registration
CLIENT_ID = os.getenv('CLIENT_ID')
# Application's generated client secret: never check this into source control!
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

# AUTHORITY = "https://login.microsoftonline.com/common"  # For multi-tenant app
AUTHORITY = f"https://login.microsoftonline.com/{os.getenv('TENANT_ID', 'common')}"

REDIRECT_PATH = "/getAToken"  # Used for forming an absolute URL to your redirect URI.
# The absolute URL must match the redirect URI you set
# in the app's registration in the Azure portal.

# You can find more Microsoft Graph API endpoints from Graph Explorer
# https://developer.microsoft.com/en-us/graph/graph-explorer
ONEDRIVE = 'https://graph.microsoft.com/v1.0/me/drive/root/children'  # This resource requires no admin consent

# You can find the proper permission names from this document
# https://docs.microsoft.com/en-us/graph/permissions-reference
SCOPE = ["User.Read, User.ReadBasic.All, Files.Read, Files.Read.All, Files.ReadWrite, Files.ReadWrite.All, offline_access"]

# Tells the Flask-session extension to store sessions in the filesystem
SESSION_TYPE = "filesystem"
# Using the file system will not work in most production systems,
# it's better to use a database-backed session store instead.
