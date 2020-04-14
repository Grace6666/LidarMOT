# LiDAR-Based Multiple Object Tracking

**Segmentation and Object Detection**
1. Extract region of interest
2. Ground points segment via ground plane fitting
3. Obstacle segment using DBSCAN based 3D clustering
4. Bounding box estimation via L-shape fitting

**Object Tracking**
1. Kalman Filter
2. 2D Assignment based data association
3. Track management: initialize, confirm and delete tracks

![alt text](https://github.com/Grace6666/LidarMOT/diagram.png)
