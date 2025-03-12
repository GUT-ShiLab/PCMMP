# PCMMP

### 1. 数据

本研究所使用的宏基因组-代谢组配对数据集来自"The gut microbiome-metabolome dataset collection: a curated resource for integrative meta-analysis"处理后的数据，数据获取链接为https://github.com/borenstein-lab/microbiome-metabolome-curated-data。其中，原始宏基因组测序数据经过标准质控、宿主去除及分类学分配，最终生成微生物属层级相对丰度数据；代谢组数据不做任何预处理，只是通过KEGG和HMDB等工具进行代谢物注释。

### 2. Code

（1）环境配置

|   Package    | Version |
| :----------: | :-----: |
|    numpy     | 1.26.4  |
|    pandas    |  2.2.2  |
|    python    | 3.9.19  |
|   pytorch    |  2.4.0  |
| pytorch-cuda |  11.8   |

（2）代码示例（以IBD为例）

系统发育排序的多尺度特征构建：实现该部分主要依赖三个代码文件，运行顺序和主要功能如下，最后需要将(1)(3)结果进行额外合并，构建模型输入数据。

```python
phylo_fea    		#（1）基于NCBI Taxonomy数据库获取属层级微生物的NCBI ID，利用ETE3构建系统发育树并进行后序遍历
mic_clr_process    	#（2）数据转换为CLR格式
phylo_group    		#（3）采用卷积核引导的等间隔特征选择策略从后序遍历序列中提取全局关联特征
```

卷积网络的特征学习与代谢丰度预测：直接执行main.py代码文件即可，各个模块分别放在了不同的文件夹下便于用户理解。

```python
python main.py
```

结果绘制：绘图代码放置在utils文件夹下，每个代码主要实现的功能如下。

```python
plot_dims_search    #绘制网络参数的消融实验柱状图
plot_pca    		#绘制IBD跨队列实验的数据PCA图
plot_result_test    #绘制IBD跨队列实验的方法对比图
plot_result_xr      #绘制不同模型和数据的消融实验图
```

