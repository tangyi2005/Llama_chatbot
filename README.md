## 简介
这是基于Llama3进行4bit量化微调后的模型开发的一个聊天机器人。
其特点占用空间小，能在机子为4GPU快速运行起来。是一个很好学习工具。

## 安装依赖
pip install transformers
pip install streamlit
pip install torch
pip install modelscope

## 下载模型
进入modelscope的https://modelscope.cn/models/ty200509/Llama-3.1-8B-bnb-4bit下载模型
也可以使用一下方式下载：
```python
from modelscope import snapshot_download
model_dir = snapshot_download("ty200509/Llama-3.1-8B-bnb-4bit",cache_dir="下载模型到本地路径")
```

## 下载代码
代码地址：https://github.com/tangyi2005/Llama_chatbot.git
使用命令进来Llama_chatbot项目的llama_chat.py
执行以下命令：
```
# streamlit run llama_chat.py所在文件的路径如:D:\XXXX\XXXX --server.address 127.0.0.1 --server.port 6006
```




