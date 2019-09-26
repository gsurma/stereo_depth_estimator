<h3 align="center">
  <img src="assets/stereo_depth_estimator_icon_web.png" width="300">
</h3>

# Stereo Depth Estimator

![](assets/stereo_setup.png)


## SGBM
**Semi-global block matching** - SGBM, by Hirschmuller, H. (2008). Stereo processing by semiglobal matching and mutual information. IEEE Transactions on pattern analysis and machine intelligence, 30(2), 328-341.

### Usage

`python3 stereo_depth_estimator_sgbm.py`


Required directory structure:

    .
    ├── ...
    ├── data                     
    │    ├── 1 (can be modified with DATASET constant)
    │        │──left (n left images)
    │        │──right (n right images)
    │        │──disparities (initially empty)
    │        │──combined (initially empty)



### Results

#### Dataset 1
![](data/1/result.gif)

#### Dataset 2
![](data/2/result.gif)

#### Dataset 3
![](data/3/result.gif)



## Author

**Greg (Grzegorz) Surma**

[**PORTFOLIO**](https://gsurma.github.io)

[**GITHUB**](https://github.com/gsurma)

[**BLOG**](https://medium.com/@gsurma)

<a href="https://www.paypal.com/paypalme2/grzegorzsurma115">
  <img alt="Support via PayPal" src="https://cdn.rawgit.com/twolfson/paypal-github-button/1.0.0/dist/button.svg"/>
</a>

