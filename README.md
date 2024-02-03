![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
<p align='center'>
    <h1 align="center">Traffic detection</h1>
    <p align="center">
    Project for Signal, Image and Video at the University of Trento A.Y.2023/2024
    </p>
    <p align='center'>
    Developed by:<br>
    De Martini Davide <br>
    Mascherin Matteo <br>
    </p>   
</p>

----------

- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the project](#running-the-project)


## Project Description
For this project, our goal is to develop software capable of
real-time traffic detection. Given a video input (e.g. a video
of a security camera of a highway), the software will identify
the street, crop it, and pinpoint vehicles by highlighting their
bounding boxes along with their respective directions. To
simulate a real time video flow, an HTTP server takes the
input video, process it and streams the resulting frames to
the connected clients. The video stream quality is selected
by the client according to the network capabilities.

## Project Structure
``` BASH
Desktop/traffic_detection
├── assets
│   ├── moving_cam.mp4
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── config.py
├── counting_vehicles.py
├── http_streaming_server.py
├── README.md
├── requirements.txt
├── sockets
│   ├── receiver.py
│   └── sender.py
└── templates
    ├── index.html
    └── video_viewer.html

```

`http_streaming_server.py` is the file for starting the webserver. This will expose two API for the client to connect to the server and receive the video stream.


`counting_vehicles.py` is the file to run directly a simulation of the traffic detection. It will read the video from the assets folder and process it. It also contains the code for the traffic detection used in the server.

## Installation

In order to run the project you'll need to clone it and install the requirements. We suggest you to create a virtual environment 
- Clone it

    ```BASH
    git clone https://github.com/davidedema/traffic_detection

    ```
- Create the virtual environment where you want, activate it and install the dependencies 
  
    ```BASH
    cd path/of/the/project
    python -m venv /name/of/virtual/env
    source name/bin/activate
    pip install -r requirements.txt
    ```

## Running the project

The project could be runned in two different ways:
- Through web server
  
    ```
    python http_streaming_server.py
    ```
    After running the server, the url is shown directly in the terminal. You can connect to the server by typing the url in the browser. If you want to watch the stream from a media player such as VLC, you can simply execute the following command:

    ```BASH
    vlc servel_url/stream_video
    ```

- Running directly:
  
    ```
    python traffic_detection.py
    ```