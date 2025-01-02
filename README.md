# 训练数据
## 预训练数据
`medical_encyclopedia.json`
(该文件较大，超过500MB，未上传到仓库中，可自行下载)

数据来源：https://huggingface.co/datasets/shibing624/medical

使用的是`/pretrain/train_encyclopedia.json`这个文件

行数：361,420

token数：144,732,087

平均每行token数：400

## SFT数据
`MedQA_chinese_qbank.jsonl`

数据来源：https://github.com/jind11/MedQA (有给出Google Drive下载链接)

使用的是`data_clean\questions\Mainland\chinese_qbank.jsonl`这个文件

行数：34,253

### 数据集预处理

使用`train/preprocess/preprocess_MedQA.py`进行预处理，将原数据集转换为alpaca格式，得到`MedQA_chinese_qbank_alpaca.json`，方便训练时使用。

转换规则如下：

- 原数据集格式：`question`, `options`(列表), `answer`, `meta_info`
- 转换后
  - instruction：你是一个{`meta_info`}问题助手，帮助用户解答{`meta_info`}问题。请根据问题选择正确的答案，并只输出字母（A、B、C、D或E）。
  - input：`question`+所有`options`(以`\n`分隔)
  - output：`answer`

# 训练配置

# 验证
运行`validate.py`可以验证模型在validation set上的准确度，输出会保存在`output`文件夹。