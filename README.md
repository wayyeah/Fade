# Fade
Fade: Fast and Deployable 3D Object Detection for Autonomous Driving
### KITTI 3D Object Detection Baseline
|                                              | Car_R40@0.7 Easy| Car_R40@0.7 Mod. | Car_R40@0.7 Hard  | download | download(TensorRT) | 
|---------------------------------------------|:-------:|:-------:|:-------:|:---------:|:---------:|
| Fade | 90.92 | 82.00 | 77.49 | [model-50M](https://drive.google.com/file/d/1NlOdfU745UfT0ptywPhEhBhevAPdzib0/view?usp=sharing) |  [model-25M](https://drive.google.com/file/d/1Zsb3n7xR25IWWENYK1Igff6fNzSQOPgP/view?usp=sharing) | 
## Now, the code of Fade-TRT has been uploaded, the training code will be released in this repo when the paper has been accepted.
### How to use
A. prepare environment
  1. install CUDA>=11.4
  2. install TensorRT>8.5.x.x and modified TensorRT Path in fade_trt/CMakeLists.txt
  3. modified path in fade_trt/main.cpp Line 137 model_path and Line 138 path
B. compile and run
  1. ```cd fade_trt```
  2. ```mkdir build & cd build ```
  3. ```cmake .. & make ```
  4. ``` ./demo kitti ```

## We deployed our Fade object detection algorithm on a car equipped with an OS-128 LiDAR and Jetson Orin, and achieved autonomous driving of the car.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/wayyeah/Fade/blob/master/car.png" width="500">
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f03e2a0f-12ff-4b82-b4a0-178dd2e6f26c" width="180">
    </td>
    
  </tr>
</table>

Video: https://github.com/wayyeah/Fade/blob/master/AutonomousDriving.mp4
