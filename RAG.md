# <font color=#0099ff> **RAG** </font>

> `@think3r` 2026-03-20 00:28:17

- RAG (Retrieval-Augmented Generation, 即检索增强生成) : 知识库和 embding 模型 ?  deepWiki ?

![rag](./rag.drawio.svg)

| 特性 | RAG (检索增强) | Fine-tuning (微调) |
| :--- | :--- | :--- |
| **类比** | **开卷考试**（翻书找答案） | **闭卷考试**（把知识背进脑子） |
| **更新速度** | 极快（直接更新数据库即可） | 慢（需要重新训练） |
| **准确性** | 高（有出处可查，减少幻觉） | 中（仍可能产生幻觉） |
| **擅长领域** | 事实性知识、外部文档 | 特定语气、格式、复杂任务逻辑 |
| **成本** | 较低 | 较高（算力开销大） |

## <font color=#009A000> 0x01 DeepWiki </font>

- <https://deepwiki.com>
- [DeepWiki 使用方法与技术原理深度分析](https://zhuanlan.zhihu.com/p/1912652800820749046)

原理 (不保真...) :

1. 将代码源文件解析成 AST 抽象语法树结构
2. 静态分析 : 提取代码中的实体和关系。
   1. 实体包括函数、类、变量、模块等；
   2. 关系包括调用关系（如函数 A 调用函数 B）、继承关系（如类 B 继承自类 A）、依赖关系（如模块 X 导入模块 Y）等。
3. 借鉴 Graphbrain 知识建图
4. 语义融合

## <font color=#009A000> 0x02 方案 </font>

### <font color=#FF4500> 文档处理 </font>

- [unstructured](https://github.com/Unstructured-IO/unstructured)
  - 开源的, python 版本非结构化数据预处理的开源库（及服务）
  - 将各种格式乱七八糟的文档（PDF、PPT、Word、HTML 等），转化为大模型（LLM）能够理解的干净、结构化的文本块（Chunks）
  - 提供 :
    - Partitioning（分块解析）： 识别文档中的不同元素。例如，它能分清 PDF 里哪里是标题、哪里是正文、哪里是表格或图片说明。
    - Cleaning（清洗）： 自动去除文档中的“噪音”，比如多余的空格、乱码、特殊控制字符或特定的页眉页脚。
    - Extracting（提取）： 从文档中提取特定元数据（Metadata），如作者、创建日期、文件名等，这对后续的过滤检索非常有用。
    - Staging（适配）： 将处理好的数据直接转化为符合下游框架（如 LangChain、LlamaIndex 或各种向量数据库）要求的格式。
- [PyMuPDF4LLM](https://github.com/pymupdf/pymupdf4llm)
  - 开源工具 python 工具
  - 将 PDF 转换为适合 LLM 的 Markdown (专门针对 LLM 的衍生工具)。
  - 把 PDF 里复杂的布局（比如双栏、表格、页眉页脚）转换成标准的 Markdown 格式, 能保留标题层级（# ##）和表格结构，这对 LLM 理解文档逻辑至关重要。
- [RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters)
  - 根据分隔符（如 \n\n, \n,  ）递归地切分文本，直到每一块都小于你设定的 chunk_size。
