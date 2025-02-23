import identity
import identity.web
import requests
from flask import Flask, redirect, render_template, request, session, url_for
from flask_session import Session

import app_config

app = Flask(__name__)
app.config.from_object(app_config)
Session(app)

# This section is needed for url_for("foo", _external=True) to automatically
# generate http scheme when this sample is running on localhost,
# and to generate https scheme when it is deployed behind reversed proxy.
# See also https://flask.palletsprojects.com/en/2.2.x/deploying/proxy_fix/
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

auth = identity.web.Auth(
    session=session,
    authority=app.config.get("AUTHORITY"),
    client_id=app.config["CLIENT_ID"],
    client_credential=app.config["CLIENT_SECRET"],
)


@app.route("/login")
def login():
    return render_template("login.html", version=identity.__version__, **auth.log_in(
        scopes=app_config.SCOPE, # Have user consent to scopes during log-in
        redirect_uri=url_for("auth_response", _external=True), # Optional. If present, this absolute URL must match your app's redirect_uri registered in Azure Portal
        ))


@app.route(app_config.REDIRECT_PATH)
def auth_response():
    result = auth.complete_log_in(request.args)
    if "error" in result:
        return render_template("auth_error.html", result=result)
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    return redirect(auth.log_out(url_for("index", _external=True)))


@app.route("/")
def index():
    if not (app.config["CLIENT_ID"] and app.config["CLIENT_SECRET"]):
        # This check is not strictly necessary.
        # You can remove this check from your production code.
        return render_template('config_error.html')
    if not auth.get_user():
        return redirect(url_for("login"))
    return render_template('index.html', user=auth.get_user(), version=identity.__version__)


@app.route("/onedrive_list")
def call_downstream_api():
    token = auth.get_token_for_user(app_config.SCOPE)
    if "error" in token:
        return redirect(url_for("login"))
    # Use access token to call downstream api
    api_result = requests.get(
        app_config.ENDPOINT,
        headers={'Authorization': 'Bearer ' + token['access_token']},
        timeout=30,
    ).json()
    return render_template('display.html', result=api_result)

@app.route("/upload", methods=["POST"])
def upload_file():
    # Get token using identity.web's method
    token_result = auth.get_token_for_user(app_config.SCOPE)
    if "error" in token_result:
        return redirect(url_for("login"))
    
    if "file" not in request.files:
        return "No file selected", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    try:
        # Prepare upload to OneDrive
        headers = {
            "Authorization": f"Bearer {token_result['access_token']}",
            "Content-Type": "application/octet-stream"
        }

        # Properly format the filename for OneDrive URL
        filename = file.filename.replace(" ", "_")  # Sanitize filename
        upload_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{filename}:/content"

        # Stream the file content instead of reading all at once
        response = requests.put(
            upload_url,
            headers=headers,
            data=file.stream  # Use stream for better memory handling
        )

        if response.status_code in (200, 201):
            return "File uploaded successfully!"
            
        # Handle Microsoft Graph errors
        error_msg = response.json().get("error", {}).get("message", "Unknown error")
        return f"Upload failed: {error_msg}", response.status_code

    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}", 500
    except Exception as e:
        return f"Unexpected error: {str(e)}", 500

if __name__ == "__main__":
    app.run()
