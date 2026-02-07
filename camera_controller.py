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


# Example standalone usage
if __name__ == "__main__":
    cam = IPCameraController("192.0.0.4")
    result = cam.set_resolution(1280, 720)
    print(result)
