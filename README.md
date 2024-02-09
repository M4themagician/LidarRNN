# LidarRNN

A proof of concept recurrent neural network to predict dynamics on a gridmap of (low effort) synthetic 2D lidar data.

[![Predictions Visualized](https://img.youtube.com/vi/T7rDNcp-8W4/0.jpg)](https://www.youtube.com/watch?v=T7rDNcp-8W4)

## Installation

Tested with Pytorch 2.0 and up. Install from [here](https://pytorch.org/get-started/locally/).
It also needs opencv, so after setting up your python environment do a quick little 


```bash
pip install opencv-python
```

For running anything add the repos root to you pythonpath:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/this/repo
```

## Usage

It's either training a model via train.py or visualizing a trained model vie demo.py. Training has no stopping criterion for now. The demo shows the input/outputs or optionally writes it into a video.

