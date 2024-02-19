# YOLOv8--DM3-STEM-Automated-Nanoparticle-Analysis

## Description
STEM-Automated-Nanoparticle-Analysis-YOLOv8-SAM is a software designed for the automated detection and analysis of nanoparticles in STEM (Scanning Transmission Electron Microscopy) images, specifically for DM3 file formats. This tool efficiently extracts the image and the scale in nanometers per pixel to provide accurate and detailed analyses. This software is based in part of https://github.com/ArdaGen/STEM-Automated-Nanoparticle-Analysis-YOLOv8-SAM. Translated to work with DM3 file format.

## Key Features
- **Automated Extraction:** The program processes DM3 files to extract images and determine the pixel-to-nanometer conversion scale, facilitating subsequent interpretation and analysis.
- **Nanoparticle Detection:** Utilizing the advanced YOLOv8 algorithm, the system automatically identifies nanoparticles within the image, optimizing analysis precision and speed.
- **Selection and Analysis:** From all detected nanoparticles, the program selects a representative set of 500, calculating the average size of these to provide relevant and useful statistics for research and industrial applications.

## Applications
This software is essential for researchers and professionals working in nanotechnology, advanced materials, chemistry, physics, and related fields requiring detailed and accurate nanoparticle analysis.

## Contributing
We invite the community to contribute to the project by submitting issues, feature requests, and pull requests. Every contribution helps make this software a better tool for everyone.

## Models weight
Weights for YOLOv8 S/TEM nanoparticle object detection
Obtained from https://github.com/ArdaGen/STEM-Automated-Nanoparticle-Analysis-YOLOv8-SAM
https://drive.google.com/file/d/1XY8FXWtPb8T-QkiEPR0ENWRPcck_aHRS/view?usp=sharing

## Installation
Install [PyTorch](https://pytorch.org/get-started/locally/)
<br>
<br>
Install Ultralytics for YOLOv8
```
pip install ultralytics

