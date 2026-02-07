import requests

IP = "192.0.0.4"
PORT = 8080

# Force 1080p
url = f"http://{IP}:{PORT}/settings/video_size?set=1280x720"

r = requests.get(url)

print("Status:", r.status_code)
print("Response:", r.text[:200])
