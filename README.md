# Steps to Run the Application

## 1. Clone the Repository

Download or clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

---

## 2. Install Ngrok

Download and install Ngrok from: https://ngrok.com/download

---

## 3. Start Ngrok

Open **Terminal 1** and run:

```bash
ngrok http 8000
```

Keep this terminal running.

---

## 4. Configure API Keys

Open `.env` in VS Code and update it with your API keys if required.

> **Important:** Please use your own API keys. The existing keys are only for testing purposes. Update the keys in the `.env` file before running the application.

Save the file using:

```text
Ctrl + S
```

---

## 5. Start the Application Server

Open **Terminal 2**.

If the server is already running, stop it using:

```text
Ctrl + C
```

Then start the application:

```bash
python app_full.py
```

---

## 6. Create a Microsoft Teams Meeting

1. Open your personal Microsoft Teams application.
2. Click **Meet Now** to start an instant meeting.
3. Copy the meeting link.

Example:

```text
https://teams.live.com/meet/9360266452839?p=jT6iLe7CDkFlVIBJoh
```

---

## 7. Trigger the Bot

Open **Terminal 3** and run:

```bash
curl "http://127.0.0.1:8000/trigger-bot?meeting_url=<PASTE_TEAMS_MEETING_LINK_HERE>"
```

Example:

```bash
curl "http://127.0.0.1:8000/trigger-bot?meeting_url=https://teams.live.com/meet/9360266452839?p=jT6iLe7CDkFlVIBJoh"
```

Alternatively, you can open the following URL directly in your browser:

```text
http://127.0.0.1:8000/trigger-bot?meeting_url=<PASTE_TEAMS_MEETING_LINK_HERE>
```

---

## Notes

- Ensure Ngrok is running before starting the application.
- Keep all terminals open while testing.
- Use your own API keys in the `.env` file.
- Replace the example Teams meeting link with your own meeting URL.
