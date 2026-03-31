# huskyadaptCamera

Project Overview

This project aims to estimate spatial depth using a single camera combined with the MiDaS depth estimation algorithm. The system computes relative distance information from monocular vision and enhances accuracy by incorporating an ultrasonic sensor to provide reference distance measurements.

By fusing vision-based depth estimation with ultrasonic sensing, the system is designed to generate a more reliable and scalable depth perception solution, with the long-term goal of constructing a wide-range point cloud representation of the environment.

Current Progress
Implemented monocular depth estimation using MiDaS
Established Bluetooth communication between the sensing system and the response module
Built initial pipeline for real-time data transmission
Future Work
Integrate ultrasonic sensor for absolute distance reference
Synchronize data acquisition between camera and ultrasonic sensor
Perform sensor fusion to improve depth reliability and robustness
Extend system to generate large-scale point cloud data
Goal

The ultimate goal is to develop a hybrid proximity sensing system that combines vision and ultrasonic sensing to achieve higher reliability and accuracy compared to single-sensor approaches.
