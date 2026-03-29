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


## <font color=#009A000> 0x05 老黄的播客 </font>

> [Jensen Huang: NVIDIA - The $4 Trillion Company \& the AI Revolution \| Lex Fridman Podcast](https://www.youtube.com/watch?v=vif8NQcjVf0)

- 从单一 GPU 到机架的设计, 复杂性的扩散 :
  - 计算 CPU/GPU
  - 网络 : network/switch
  - 负载均衡
  - 供电/散热
- TODO:
  - HBM 供应链 ??
  - token/W
- 第一性原则 : 物理极限是什么, 当前阶段是哪个
- Ai 算力和 extra Engerge 的利用 :
  - UPS 缓冲 + 电网监测 + 实时调度 NVIDIA GPU 的 power profiles
  - 外太空的能源利用 : 算力上天.
    - 太空遥测数据的巨大数据量 <---> Ai 预处理再传回
- 关于 Ai 训练的数据 :
  - 已有高质量数据的枯竭
  - 合成数据比例的上升
- Nvidia install Base 定义的 CUDA :
  - gamming 加速器, 太窄/太专
  - 可编程 Pixel Shader
  - FP32 支持 (Cg)
  - CUDA of GeForce --> computer conmany
    - 成本的提升, 第一个生存威胁的决策
  - CUDA 的扩散, SDK 的新人性
- 计算系统的分割 :
  - 旧有计算系统 : 检索系统 --> 存储/仓库是大头
  - Ai 系统 : 生成系统 --> 计算是大头 (更像一个 token 工厂)
- 计算机视觉超过人类已经很久了.
  - 放射科医生的处境
- AGI 的两大组成
  - 智力 (功能性的工具)
  - 情感/情绪. ai 能判断情绪, 但无法真正感知它.
    - 人和人的情绪是不同的, 但两个 ai 在相同 contex 下概率学上是相近的.
- 遗忘的能力 :
  > And part of it—part of the process—is forgetting. 其中一个重要的部分，其实是“遗忘”。 <br/>
  > One of the most important attributes of AI learning, as you know, is systematic forgetting. You need to understand what to forget. You can’t memorize everything or retain all information indefinitely. 正如你所知道的，AI 学习中一个非常关键的特性就是“系统性的遗忘”（systematic forgetting）。你需要知道什么时候该忘记什么内容。你不可能记住所有东西，也不应该保留所有信息。 <br/>
  > Instead, you should avoid carrying unnecessary information. One of the things I do very quickly is decompose the problem, reason about it, and distribute the cognitive load. When I say I “tell everybody,” what I’m really doing is sharing that burden as efficiently as possible. 相反，你应该避免背负不必要的信息。我通常会很快地把问题拆解开，对问题进行推理，并分担认知负担。当我说“我告诉所有人”的时候，本质上是在尽可能快地把这个负担分散出去。


### <font color=#FF4500> ai 版本总结 </font>

1. 计算架构：从单片 GPU 向机架级集成演进
   - **复杂性扩散：** 算力设计重心已由微观芯片转向**机架级全栈集成**，涵盖 CPU/GPU 异构协同、高性能网络拓扑（Network/Switch）、动态负载均衡及热电管理系统。
   - **第一性原理：** 需回归物理本质，审视 **HBM 供应链** 弹性与 $Token/W$ 能效比的极限，界定当前技术演进的宏观边界。
2. 能源革新：智能电网调度与星际算力前瞻
   - **能源深度协同：** 通过 **UPS 动态缓冲**与电网监测，实时调度 GPU 的 **Power Profiles**（功耗特性），实现算力负荷与能源供给的毫秒级解耦。
   - **星载 AI 预处理：** 针对外太空遥测数据带宽瓶颈，采取“算力上天”策略，通过 **AI 边缘预处理**压缩冗余信息，解决跨星际通信的数据回传压力。
3. 数据范式：从原生矿藏到合成进化
   - **语料枯竭：** 互联网高质量人类原生数据的开采已接近物理极限，数据稀缺成为算力释放的主要阻碍。
   - **合成数据（Synthetic Data）：** 行业正转向以高占比合成数据驱动模型训练，标志着 AI 演进进入自进化与逻辑闭环的新阶段。
4. NVIDIA 战略路径：CUDA 的生态渗透与普适化
   - **范式转移：** 从专用图形加速器转向通用计算（GPGPU）。通过可编程 **Pixel Shader** 与 **FP32 (Cg)** 支持，完成了从“窄众工具”向“计算基座”的跨越。
   - **生态护城河：** 在 GeForce 系列强制推行 CUDA 是一次高成本的生存博弈，最终通过 **SDK 的易用性**与开发者社区的渗透，确立了无可撼动的计算标准。
5. 系统重构：从“检索检索”向“Token 工厂”转型
   - **传统计算系统：** 以**存储与 I/O** 为核心，本质是存量信息的检索、搬运与仓库式管理。
   - **AI 生成系统：** 转向以**计算**为核心，系统职能演变为高效产出逻辑与信息的 **“Token 工厂”**，计算密度取代存储容量成为衡量标准。
6. AGI 边界：功能性智力与感知模拟的差异
   - **专业领域超验：** 计算机视觉（CV）在放射影像诊断等特定专业领域已长期超越人类表现。
   - **智力与感知的解耦：** AGI 具备强大的功能性智力，但在情感层面仅停留于**逻辑判断**而非真实感知。
   - **概率趋同性：** 人类情绪具有高度个体异构性，而 AI 在相同 Context（上下文）下的输出在概率学上具有高度趋同性。

![ai 四个阶段](./image/LLM%20阶段.png)
