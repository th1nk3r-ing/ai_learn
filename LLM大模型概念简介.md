# <font color=#0099ff> **速通概念 --> LLM 文本大模型** </font>

> `@think3r` 2025-02-08 23:50:03
>
> - [安德烈·卡帕西最新AI普及课：深入探索像ChatGPT这样的大语言模型 Andrej Karpathy](https://www.bilibili.com/video/BV1WnNHeqEFK)
>   - ori : <https://www.youtube.com/watch?v=7xTGNNLPyMI>
> - [🍷 FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

## <font color=#009A000> 0x00 简介 </font>

<https://drive.google.com/file/d/1EZh5hNDzxMMy05uLhVryk061QYQGTxiN/view>

![llvmIntro](./image/LLMIntro-2024-07-22-1743.svg)

## <font color=#009A000> 0x01 预训练 pre-trainning </font>

1. 数据集处理, E.g. : [huggingFace-FineWeb 数据集](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).
   - --> 过滤后的最终文本数据 `44TB`
   - 目的 : 高质量, 多样性, 数据量大
   - CommonCrawl : 自 2007 年开始的网络爬虫组织
     - url-filtering : <https://dsi.ut-capitole.fr/blacklists/>
   - text-extraction : 去除 html 标记
   - lang-filtering : 保留 85% 以上英文的网页
   - ... 重复信息删除
   - PII remove : personally identifiable information remove 个人信息移除
   - 最终处理结果 : <https://huggingface.co/datasets/HuggingFaceFW/fineweb?row=0> -> Dataset Preview 部分
2. tokenization (一维) 分词 :
   - Converts text <---> sequences of symbols (/tokens) 尽可能的压缩文本数据量
     - why ? 输入的 token 越多/越长, 这个神经网络的 **正向传播**(`forward pass` / `Forward Propagation`) 就越昂贵(~~越耗费计算资源~~).
       - TODO:
         - [怎样理解tensorflow的正向和反向传播?](https://www.zhihu.com/question/395116009)
         - [Back Propagation（梯度反向传播）实例讲解](https://zhuanlan.zhihu.com/p/40378224)
     - GPT4 使用 `100277` 个符号
   - <https://tiktokenizer.vercel.app/>

### <font color=#FF4500> neural network training </font>

1. 窗口上下文数量 maximum-context-length
2. 输入 sequence of tokens --> trained-model --> 让下一个正确序列的概率更高.
   1. 模型 : 神经网络的参数/权重
   2. 神经网络结构 : python 代码 `model.py` (k loc 级别, 并不复杂), 描述了模型中的操作步骤序列
   - 模型的发布最少需要上述两者
3. LLM Visualization : <https://bbycroft.net/llm>
4. `GPT` (Generatively Pre trained Transformer) :
   - 论文 : [**attention is all you need**](https://arxiv.org/pdf/1706.03762), 2017 By Google
   - TODO: 论文阅读工具...
   - [GPT2 - 2019](https://github.com/openai/gpt-2.git) :
     - > first time that a recognizably modern stack came together : 首次出现一个可被明确视为现代技术栈的组合
     - 1.6 billion parameters
     - maximum context length of 1024 tokens
     - trained on about 100 billion tokens
     - 论文 : <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
     - [论文阅读：Language Models are Unsupervised Multitask Learners](https://zhuanlan.zhihu.com/p/711058430)
   - 复现项目 : `LLM.c` --> <https://github.com/karpathy/LLM.c/discussions/677>
      - loss : 单个浮点数, 衡量神经网络当前的性能, 越小越好.
      > 在机器学习，尤其是深度学习中，损失函数（loss function）是一个核心概念，它用于衡量模型预测值与实际目标值之间的不一致程度。Loss值越小，表示模型的预测结果越接近真实值，模型性能越好。
   - llama-3 :
     - 论文 : [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)

### <font color=#FF4500> 基础模型 base-model </font>

- 分类
  1. base-model 基础模型 : 互联网文本模拟器
  2. instruct-model 指令模型
- <u>模型本质上是对训练数据(互联网数据)的一种有损压缩(大致印象), 而对话则是将其中的某些部分进行回忆</u>
  - 训练数据是落后与真实世界的互联网的, 而模型的回答只是对 **历史** 互联网数据的 **模糊** 回忆, 因此其回答 <u>不可信</u>
  - 当你复制 wikipedia 的内容来提问时, 模型可能输出和 wiki 完全一致的回答
    - 反刍 regurgitation : 直接引用已训练过的数据
    - 原因 : Wikipedia 被认为是高质量数据, 可能有较高的训练优先级(训练/看过 10 次, 以至于模型像人一样完全记住了, 数据记忆的过于准确了)
- 提示 prompt
  - in-context learning 上下文学习 --> 解决未知问题的一部分能力. E.g. : 单词翻译的序列
  - (基础模型)的提示词 prompt 的撰写/优化

## <font color=#009A000> 0x02 后训练(post-trainning) 1 : SFT 监督微调 </font>

- post-trainning 后训练 : supervised fine-tuning 监督微调 (`SFT`) --> 个性的出现
  - 将基础模型转换为一个回答助手
- human labelers (人工标注者) 手动回答问题, 用于模型后训练
  - 数据集很小, 因此训练时间会很短
- 对话的分词 tokenization of Conversation :
  - tokenization 的 encoding : 见上图的 `Conversation Protocol / Format`
  - NOTE: `<|im_start|>` 是一个标签/分隔符, 基础模型并不包含, 其是新增的, 目的是为了进行对话训练
- `InstructGPT` 论文 : [Training language models to fol instructions with human feedback](https://arxiv.org/pdf/2203.02155)
  - 人工标注回答的指导文档通常有几十页之多
  - 开源的标注集 : [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)
  - 现在很少有人手动从头编写, 多使用 LLM 直接生成答案, 然后编辑/修改它
    - 原因在于难度 : `创作 >> 修改`
    - SFT 数据集的一个例子 : [ultraChat](https://github.com/thunlp/UltraChat)
      - 涵盖的方面 : <https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac>
    - 某些特定的技术/领域需要对应专家的参与标注集的确认, E.g. : code, 医药

### <font color=#FF4500> LLM psyhology 心理学 </font>

- `Hallucinations` **幻觉** : 完全虚构信息
  - 消除方法 :
    1. > Use model interrogation to discover model's knowledge, and programmatically augment its training dataset with knowledge-based refusals in cases where the model doesn't know
       - 利用模型询问来发掘模型的知识，并在模型不知情的情况下，通过编程方式用基于知识的拒绝来扩充其训练数据集
       - 用新的未训练过的知识(有上下文)询问已训练好的模型, 得出问题&答案(利用模型上下文推理能力), 然后将问题其加入训练集 : 网络中某个神经元值很高时, 让模型输出不知道 (模型本身是有这个能力的, 我们只需加强这种能力, 并将其表达出来)
    2. > Allow the model to search!
       - `<SEARCH_START>` 引入的新 token, 搜索引擎搜索, 爬虫, 进入上下文窗口(隐形), 后续的回答就进入了下一个 context-window (工作内存), 而该状态中包含了搜索的信息
  > - `Knowledge in the parameters == Vague recollection` (e.g. of something you read 1 month ago) (参数中的知识 == 模糊的回忆（例如 1 个月前读过的东西）)
  > - `Knowledge in the tokens of the  context window == Working memory`  (上下文窗口符号中的知识==工作记忆)
- 模型的自我认知 Knowledge of self
  - 询问模型是谁, 本质上受是训练集数据影响的, 而各种开源数据集中都有 chat-GPT/其它模型的影子
    - E.g. : [allenai/olmo-2-hard-coded](https://huggingface.co/datasets/allenai/olmo-2-hard-coded) 开源训练集
  - 另一个原因是 `system` 系统信息提示词, 它存在于 context-window 中, 也可以影响 llm 的认知
- Models need tokens to think
  - 模型下一个 tokens 的生成中计算量(资源)是有限的, 不可能一下子算出复杂的结果
  - 如果可以, 让模型使用工具(code), 而不是让模型将其全部内容都存储在 memory 中
- 奶酪模型 : 大部分情况下都很好, 但某些独特案例中几乎会随机失败
  - 模型不擅长计数 : 计算字符串中某个字符的个数. E.g. : 虽然模型都能解决奥林匹克级别的数学问题了
    1. >但计算 `strawberry` 单词中的 r 依然会出错
       - Models are not good with spelling --> tokenization 点副作用, 模型看到的并不是单词
         - tokenization 的根源是为了减少计算量, 直接输入字符将会是很大的计算量
         - *很多人都希望完全删除 token, 转到字符/字节级的模型*
    2. > `9.11` 和 `9.9` 哪个数比较大
       - 神经网络处理这个问题时, 激活的神经元和圣经经文相关 ??? (经文的章节排序 ? `(ˇˍˇ) ～`)
       - TODO: 论文
  - 中文在这方面是还不是会更好 ?  `(ˇˍˇ) ～`
- Bunch of other small random stuff

## <font color=#009A000> 0x03 reinforcement-learning (`RL`) 强化学习 </font>

### <font color=#FF4500>  RL 简介 </font>

- `RL` 依旧属于 post-training 的范畴
- `RL` 在 llm 中新出现的阶段, 还不是很成熟, 尚未有统一的标准
- 但 `RL` 是对于 AI 领域并不是新领域 : deepMind 的 Alpho-Go 就已使用了 `RL` 训练
  - 论文 : [Mastering the game of Go without human knowledge](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/nature-go.pdf)
    - `SFT` 的上线是人类
    - `RL` 可以超越人类 : alpha-go 自我对弈, 然后自我学习
- openAi 并未开源其 RL方案
- **deepseek** 的 R1 强化学习论文及其开源的重要性 --> 重新激发公众对使用 RL 训练 LLM 的兴趣
  - 论文 : [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948) : 如何激发模型的许多推理能力
    - 模型输出的回答越来越长, 其正在学习创造非常长的解决方案
      1. 思考过程中出现的从不同角度, 追溯, 重新表达, 倒退
      2. 模型增在发现思考的方式/认知策略(cognitive strategies)
  - TODO: [论文解读DeepSeek-R1](https://zhuanlan.zhihu.com/p/19650946134)

### <font color=#FF4500> RL 原理  </font>

- why ? --> 类似于人的上学, 书籍中的主要信息和人类的学习过程

  | 序号 | 书籍学习 | llm 学习 | eng |
  | --- | --- | --- | --- |
  | 1. | 书上大部分的文字为背景阐述 | 类比预训练过程 | background knowledge `<-->` pretraining |
  | 2. | 参考问题及其解答 | 类比 SFT 用专家的知识微调 | worked problems `<-->` supervised finetuning |
  | 3. | 尝试课后练习问题(问题 + 一个最终的答案) | 利用已有的知识不断尝试/模仿 (推理), 得到最终答案, 即 `RL` | practice problems `<-->` reinforcement learning |

- 方案 : 将问题多次询问模型, 过滤其中正确的, 将其塞入作为未来的训练集中进行强化训练.

### <font color=#FF4500> 思考模型(推理模型) thinking-model </font>

- openAi 系列模型的区分 :
  - **gpt-4O 是 SFT 模型, 基本没有进行 RL 训练(做了 RLHF)**
  - O1, O3 的系列模型才是思考模型(用了 RL 训练)
- 推理模型的适用场景 :
  - 更适合于带有逻辑性推理的问题和场景
  - 一般性的知识性问题, 使用推理模型可能多此一举

### <font color=#FF4500> 不可验证领域的 RL </font>

- E.g. : 幽默感, 读摘要, 诗歌创作, 创意写作
- 论文 `RLHF` (Reinforcement Learning from Human Feedback) --> [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593) (from openAI)
  - 训练另一个完全独立奖励模型, 对被训练模型的结果进行评分/排序
    - 对人类来说, 评价比创造简单, 排序比评分要简单
- > We can run RL, in arbitrary domains! (even the unverifiable ones) This (empirically) improves the performance of the model, possibly due to the "discriminator - generator gap": In many cases, it is much easier to discriminate than to generate.  我们可以在任意域运行强化学习！（即使是无法验证的）这（经验上）提高了模型的性能，可能是由于“判别器-生成器差距”：在许多情况下，判别比生成容易得多。
- RLHF 缺点 docnside :
  - > We are doing RL with respect to a lossy simulation of humans. It might be misleading!
  - > RL discovers ways to "game" the model. It discovers "adversarial examples" of the reward model.
    - 前期效果好, 后期有很多对抗性样本(adversarial examples)
      - E.g. : `the the the the` 被认为是最好笑的笑话
  - 一般只运行几百次, 然后要剪枝/停止
- NOTE: 将小的专业的模型, 以 RL 的方式高效的塞到大的模型中 ?
- 总结 : RLHF 并不完全等价于 RL, RLHF 不能永远运行, 但是 RL 可以(其有个明确的判断标准)

## <font color=#009A000> 0x04 future 未来的演化 </font>

- **多模态模型** multimodal (not just `text` but `audio`, `images`, `video`, natural conversations)
- `tasks` -> `agents` (long, coherent, error-correcting contexts)
  - 工厂的自动化率 --> 数字世界中人工的监督
- pervasive 普遍, invisible 隐形
- computer-using
- test-time training(实时学习)?, etc
  - context-window 资源的宝贵
- [lmarena.ai](https://lmarena.ai/) : LLM 排行榜(人工)
- `LMStudio` 本地运行模型
