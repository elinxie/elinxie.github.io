
# Project 5
## Depth refocusing
```def make_refocus_gif(dir_name, c_start, c_end, step=0.05)```

To make a gif, to change your focus value from one value to another, put in the directory name of the lightfield images you want to use (make sure the only files in this directory are pictures, you can have different folders inside this directory however). The variables c_start and c_end are depth c values from -1.0 to 1.0. The function makes refocused images starting from the c_start value and going to the c_end value in steps of variable step. The default value is 0.05. The images are then put together as a gif.

## Apeture Adujustment
```def change_aperture(dir_name, start=0, end=8, c=0, lt_field=None)```

To make a gif, to change your focus value from one value to another, put in the directory name of the lightfield images you want to use (make sure the only files in this directory are pictures, you can have different folders inside this directory however). The variables start and end are apeture n values from 0 to 8 (assuming your lightfield is a 17x17 matrix of images). The function makes refocused images starting from the c_start value and going to the c_end value in increments of 1. The variable c is the focus value we set all images in the gif. It works in the same way as in `make_refocus_gif`
