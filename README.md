# conda:
```
conda create --name fasterAutoFPS python=3.9.5

conda activate fasterAutoFPS

cd C:\dev\fasterAutoFPS
```
# requirements.txt
```
pip install -r requirements.txt
```

# gpu setup paddle:
```
nvidia-smi
nvcc --version
(11.2)
conda install paddlepaddle-gpu==2.2.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/INSTALL_cn.md
https://www.paddlepaddle.org.cn/documentation/docs/zh/install/conda/windows-conda.html

# or cpu setup paddle:
```
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
python
>>> import paddle
>>> paddle.utils.run_check()
>>> exit()
python -c "import paddle; print(paddle.__version__)"
(2.2.0)
```

# setup PaddleDetection(as git submodule(of course you pull it to local with `git submodule update --recursive`))

for pycocotools:

https://visualstudio.microsoft.com/ja/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16

check only "desktop development with c++" and install
```
pip install Cython
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

cd PaddleDetection
pip install -r requirements.txt
python setup.py install

python ppdet/modeling/tests/test_architectures.py
(ok)

```
# predict
```
python PaddleDetection\deploy\python\det_keypoint_unite_infer.py --det_model_dir=C:\dev\fasterAutoFPS\TinyPose\picodet_s_320_pedestrian --keypoint_model_dir=TinyPose\tinypose_256x192 --image_file=TinyPose\example_imgs\a.jpeg --device=CPU --output_dir=TinyPose\output_dir\ --save_res=True
```



