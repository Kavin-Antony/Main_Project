import requests


class IPCameraController:
    def __init__(self, ip: str, port: int = 8080):
        self.base_url = f"http://{ip}:{port}"

    def set_resolution(self, width: int, height: int):
        res = f"{width}x{height}"
        url = f"{self.base_url}/settings/video_size?set={res}"

        r = requests.get(url, timeout=5)

        return {
            "resolution": res,
            "status": r.status_code,
            "response": r.text[:120]
        }

    def set_quality(self, quality: int):
        # Safety clamp: IP Webcam accepts 0–100, we behave like adults
        quality = max(0, min(100, quality))

        url = f"{self.base_url}/settings/jpeg_quality?set={quality}"

        r = requests.get(url, timeout=5)

        return {
            "quality": quality,
            "status": r.status_code,
            "response": r.text[:120]
        }

