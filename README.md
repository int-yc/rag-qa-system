RAG 智能问答助手

一个基于检索增强生成（RAG）的文档问答系统：把本地文档建立索引，然后基于文档内容回答用户问题。

功能
支持读取 `data` 目录下的 **TXT / PDF / Markdown** 文档

文档自动切分 + 向量化，构建 FAISS 向量索引

语义检索：理解 "语义相近" 而非仅仅字面匹配

结合大模型（豆包 API）基于文档内容生成准确回答

Streamlit 网页交互界面

技术栈

Python・LangChain (可选)・FAISS・sentence-transformers (bge-small-zh)・Streamlit・豆包 API

项目结构



```
rag\_project/

├── config/settings.py   # 配置文件（API Key、模型、参数）

├── core/

│   ├── loader.py        # 文档加载 + 文本切分

│   ├── retriever.py     # 向量化 + FAISS 检索

│   ├── llm.py           # 大模型 API 调用

│   └── rag.py           # RAG 核心流程（串联）

├── utils/logger.py      # 日志 + 装饰器工具

├── data/                # 放你的文档（txt/pdf/md）

├── app.py               # Streamlit 入口

└── requirements.txt     # 依赖清单
```

快速开始



1. 配置 API Key：编辑 `config/settings.py`，把 `api_key` 换成用户的 API Key

2. 放入文档：把想检索的文档放进 `data/` 目录

3. 安装依赖（已建好的 conda 环境可跳过）：



```
conda activate rag\_project

pip install -r requirements.txt
```



1. 启动网页：



```
streamlit run app.py
```

浏览器打开 [http://localhost:8501](http://localhost:8501)，先点【重新构建知识库】，再开始提问。

## 命令行测试（不用网页）



```
python core/rag.py
```

会进入问答循环，输入问题回车即可，输入 `exit` 退出。

国内网络配置（必看）

1. 下载向量模型（HuggingFace 被墙）

首次构建索引需要从 HuggingFace 下载中文向量模型 `bge-small-zh`，国内直连会超时。

在命令行先设置镜像再运行：



```
\# Windows (cmd/PowerShell)

set HF\_ENDPOINT=https://hf-mirror.com

python test\_retrieval.py
```

模型下载一次后会缓存到本地，之后启动不再需要联网。

2. 修改 pip 源（可选，加快依赖安装）



```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

验证检索是否正常（无需 API Key）



```
python test\_retrieval.py
```

会依次测试：文档加载 → 切分 → 建索引 → 语义检索。看到 `✅ 检索链路测试通过！` 即正常。

