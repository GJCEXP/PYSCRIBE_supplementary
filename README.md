# PYSCRIBE (supplementary)
This is a supplementary experiment for [PYSCRIBE model](https://github.com/SMAT-Lab/PyScribe).

# Runtime Environment
Runtime Environment in our experiments:
- 4 NVIDIA 2080 Ti GPUs
- Ubuntu >=16.04
- CUDA >=10.0 (with CuDNN of the corresponding version)
- Anaconda
    * Python >=3.7 (base environment)
    * Python 2.7 (virtual environment named as python27)
- PyTorch >=1.2.0 for Python 3.x
- Specifically, install our package with ```pip install my-lib-0.0.8.tar.gz``` for both Python 3.x and Python 2.7. The package can be downloaded from [Google Drive](https://drive.google.com/file/d/1YT1THhzycUF4tjnMfW4OyTpjEORLK_tO/view?usp=sharing)

# Dataset
we provide the supplementary experiment on two datasets including a Python dataset[1] and a Java dataset[2].
The whole datasets of Python and Java can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Xdk3QLQmRK7ogHCV2ZlxMYaMBlONlZb1?usp=sharing).

# Experiment on the Python Dataset
1. Step into the directory `src_code/python/`:
    ```angular2html
    cd src_code/python
    ```
2. Proprecess the train/valid/test data:
   ```angular2html
   python s1_preprocessor.py
   conda activate python27
   python s2_preprocessor_py27.py
   conda deactivate
   python s3_preprocessor.py
    ```
3. Run the model for training and testing:
   python s4_model.py
  
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path `data/python/result/code2text_4_4_4_512.json`, with ground truth and code involved for comparison.  

We have provided the results of test dataset, you can get the evaluation results directly by running 
```angular2html
python s5_eval_res.py"
```

**Note that:** 
- all the parameters are set in `src_code/python/config.py` and `src_code/python/config_py27.py`.
- If the model has been trained, you can set the parameter "train_mode" in line 83 in `config.py` to "False". Then you can predict the test data directly by using the model that has been saved in `data/python/model/`.  

# Experiment on the Java Dataset
1. Step into the directory `src_code/java/`:
    ```angular2html
    cd src_code/java
    ```
2. Proprecess the train/valid/test data:
   ```angular2html
   python s1_preprocessor.py
    ```
3. Run the model for training and testing:
   ```angular2html
   python s2_model.py
   ```
   
  
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path `data/java/result/code2text_4_4_4_512.json`, with ground truth and code involved for comparison.  

We have provided the results of test dataset, you can get the evaluation results directly by running 
```angular2html
python s3_eval_res.py"
```

**Note that:** 
- all the parameters are set in `src_code/java/config.py`.
- If the model has been trained, you can set the parameter "train_mode" in line 117 in `config.py` to "False". Then you can predict the test data directly by using the model that has been saved in `data/java/model/`.

<font size=2>[1] Wan, Y., Zhao, Z., Yang, M., Xu, G., Ying, H., Wu, J., Yu, P.S.: Improving automatic source code summarization via deep reinforcement learning. In: Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering, pp. 397–407 (2018).

[2] Hu, X., Li, G., Xia, X., Lo, D., Lu, S., Jin, Z.: Summarizing source code with transferred api knowledge. IJCAI’18, pp. 2269–2275 (2018).</font>

***This work is still under review, please do not distribute it.***