from flask import Flask, render_template, Response, request
import cv2
import threading
import time

app = Flask(__name__)

cams = {
    "cam1": {"url": "", "record": False},
    "cam2": {"url": "", "record": False},
    "cam3": {"url": "", "record": False},
}

writers = {}


def stream(cam):
    src = cams[cam]["url"]
    cap = cv2.VideoCapture(src)

    writer = None
    last_size = None

    while True:
        ok, frame = cap.read()

        # Reconnect if stream drops
        if not ok:
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(src)
            continue

        h, w = frame.shape[:2]
        size = (w, h)

        # Resolution changed -> reset writer
        if size != last_size:
            last_size = size
            if writer:
                writer.release()
                writer = None

        # Recording
        if cams[cam]["record"]:
            if writer is None:
                fname = f"{cam}_{int(time.time())}.avi"
                writer = cv2.VideoWriter(
                    fname,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    20,
                    size
                )
            writer.write(frame)

        _, jpg = cv2.imencode(".jpg", frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpg.tobytes() +
               b'\r\n')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cams["cam1"]["url"] = request.form["cam1"]
        cams["cam2"]["url"] = request.form["cam2"]
        cams["cam3"]["url"] = request.form["cam3"]
    return render_template("index.html")


@app.route("/video/<cam>")
def video(cam):
    return Response(stream(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/record/<cam>")
def record(cam):
    cams[cam]["record"] = not cams[cam]["record"]
    return f"{cam} recording: {cams[cam]['record']}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)