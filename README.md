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
├── counting_vehicles.py
├── http_streaming_server.py
├── README.md
├── receiver.py
├── requirements.txt
├── sender.py
└── templates
    ├── index.html
    └── video_viewer.html
```


The main folder is:
- `prolog_project` it contains the ROS node (motion node and **planner** node)
    - scripts: Contains the two node plus the utilities
    - msg: Contains the `.msg` file for ROS communication

`block_world.pl` is the prolog file, the core of this project.

`python_node_poc.py` is a simple proof of concept for the pyswip wrapper for prolog

## Installation