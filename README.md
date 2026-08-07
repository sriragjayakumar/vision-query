# Vision Query

Vision Query searches videos for objects and saves matching frames as images.

The project supports two detection workflows:

* Text prompt detection with YOLO-World.
* Uploaded reference image detection with YOLOE visual prompting.

It works with local video files, YouTube URLs, stream URLs, and live camera-style feeds supported by VidGear.

---

## Features

* Detect a user-specified object from a text prompt.
* Detect whether an uploaded reference image appears in a video using YOLOE.
* Supports local videos, YouTube videos, and live streams.
* Saves annotated detection frames into `detectedImages/`.
* Creates a ZIP download of extracted detection images in the Streamlit app.
* Uses CUDA automatically when PyTorch detects a compatible GPU.

---

## Apps And Scripts

### `app.py`

Streamlit web app for text-prompt object detection with YOLO-World.

Run:

```bash
streamlit run app.py
```

### `appAI.py`

Streamlit web app with two detection modes:

* `Text object prompt`: enter an object name, such as `helmet`, `person`, or `red car`.
* `Uploaded image prompt`: upload an image of the object/person/item you want to find in the video. YOLOE uses that image as the visual prompt.

The uploaded image mode uses the full reference image by default. If the uploaded image contains extra background or multiple objects, open `Reference region` and enter a tighter bounding box around the target.

Run:

```bash
streamlit run appAI.py
```

### `main.py`

Command-line YOLO-World workflow.

Run:

```bash
python main.py
```

### `mainAI.py`

Command-line YOLOE workflow.

Run:

```bash
python mainAI.py
```

---

## How It Works

1. Open a local video, YouTube URL, or stream URL.
2. Choose either a text object prompt or an uploaded reference image.
3. Run inference on every Nth frame.
4. Save annotated frames when detections are found.
5. Download the saved detections as a ZIP file from the Streamlit apps.

---

## Supported Video Sources

### Local Video

```text
videos/sample.mp4
```

### YouTube

```text
https://www.youtube.com/watch?v=XXXXXXXXXXX
```

### Live Streams

Examples include:

* RTSP cameras
* HTTP streams
* IP cameras

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/vision-query.git
cd vision-query
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Python 3.11 or newer is recommended.

---

## Models Used

### YOLO-World

Used for open-vocabulary text-prompt object detection.

Examples of searchable objects include:

* person
* helmet
* keyboard
* bottle
* laptop
* microphone
* backpack
* bicycle
* forklift
* traffic cone

### YOLOE

Used by `appAI.py` for visual prompt detection from an uploaded reference image.

The app expects the YOLOE model file:

```text
yoloe-26l-seg.pt
```

The YOLO-World app expects:

```text
yolov8x-worldv2.pt
```

---

## Project Structure

```text
vision-query/
|-- app.py
|-- appAI.py
|-- main.py
|-- mainAI.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- yoloe-26l-seg.pt
|-- yolov8x-worldv2.pt
`-- detectedImages/
```

---

## Requirements

* Python 3.11+
* Ultralytics
* PyTorch
* OpenCV
* Streamlit
* VidGear
* yt-dlp

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Future Roadmap

* Video indexing
* Timestamp extraction
* Export detections to JSON
* REST API
* Object tracking across frames
* Semantic video search

---

## License

This project is licensed under the Apache 2.0 License.
