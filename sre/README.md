# D-vector tensorflow demo, 实现方法：3DCNN + GRU
该Demo利用3D卷积神经网络和GRU神经网络实现了基于D-vector的声纹识别算法
-------------------------------
### Author: day9011
### E-mail: 329723954@qq.com  day9011@gmail.com

----------------------------
## 主要代码介绍
------------------------------------------
### make_data.py
作用：提取语音特征  
1.需要先使用sox将语音数据转为wav格式  
2.将wav格式语音数据使用libfvad做一次VAD  
3.使用kaldi-egs/local_sh/my_wav_data_prepare.sh脚本做数据准备，wav语音文件的上级目录为speaker_id  
4.使用make_data.py脚本和第三步得到的wav.scp文件提取特征数据文件name.h5  
notice：由于speaker_id会在make_data.py中会转化为number，所以要确认enroll数据集和test数据集的speaker_id一致

-----------------------------------------------------------
### sre_dvector.py
作用：训练模型  
1.修改指定路径和参数  
2.small_dataset_test是用来做小鼠聚集测试的，使用数据量为batch_size * num_small_test_batch_size  
3.num_test_utt为测试集使用的utterance数量，分段后每段语音都当作一个speaker的utterance来互相做注册验证的测试  
notice:每次运行sre_dvector.py都会清空模型存放的文件夹，所以训练新模型要更换模型存放路径

-----------------------------------------
### re_train.py
作用：从存储的模型恢复model，调整learning rate重新训练  
1.修改模型路径  
2.修改checkpoint  
3.修改learning rate, 这里我用的boundaries的方式，需要在main()的graph中修改  

---------------------------------
### valid_dvector.py
作用：使用训练集的前20个utterance做声纹验证测试  
1.修改模型路径  
2.修改checkpoint  
3.修改训练数据路径  
4.修改num_speaker与训练模型时的num_speaker一致  

--------------------------------
### evalution_dvector.py
作用：测试集数据做验证  
1.修改模型路径  
2.修改checkpoint  
3.修改num_speaker  
4,修改注册语音和测试语音数据路径  
