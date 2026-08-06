# Vision Query

**Vision Query** is a Python library that uses **YOLO-World** to detect, locate, and search for objects in local videos, YouTube videos, and live streams.

Instead of detecting every object in a frame, Vision Query searches for a **single user-specified object**, making inference faster and better suited for video search applications.

---

## Features

* 🎯 Detect a single object specified by the user.
* 📺 Supports:

  * Local video files
  * YouTube videos
  * Live streams
* 🧠 Open-vocabulary object detection using **YOLO-World**
* ⚡ Automatic GPU acceleration when CUDA is available
* 📦 Simple Python implementation with minimal dependencies

---

## How It Works

1. Open a video source.
2. Ask the user which object to detect.
3. Configure YOLO-World to search only for that object.
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

Python **3.11+** is recommended.

---

## Usage

Run:

```bash
python main.py
```

Example:

```text
Enter a YouTube URL, stream URL, or local video path:
https://www.youtube.com/watch?v=XXXXXXXXXXX

Enter the object you want to detect:
helmet
```

Or:

```text
Enter the object you want to detect:
hand
```

Vision Query automatically enables pose estimation for supported body-part queries.

---

## Project Structure

```text
vision-query/
│
├── main.py
├── utils.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Models Used

### YOLO-World

Used for open-vocabulary object detection.

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

…and many more.

---

---

## Requirements

* Python 3.11+
* Ultralytics
* PyTorch
* OpenCV
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
* Web interface
* Object tracking across frames
* Semantic video search

---

## License

This project is licensed under the Apache 2.0 License.
