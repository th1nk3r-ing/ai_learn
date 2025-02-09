# <font color=#0099ff> **速通概念 --> LLM 文本大模型** </font>

> `@think3r` 2025-02-08 23:50:03
>
> - [安德烈·卡帕西最新AI普及课：深入探索像ChatGPT这样的大语言模型 Andrej Karpathy](https://www.bilibili.com/video/BV1WnNHeqEFK)
>   - ori : <https://www.youtube.com/watch?v=7xTGNNLPyMI>
> - [🍷 FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

## <font color=#009A000> 0x00 简介 </font>

<https://drive.google.com/file/d/1EZh5hNDzxMMy05uLhVryk061QYQGTxiN/view>

![llvmIntro](./image/LLMIntro-2024-07-22-1743.svg)

## <font color=#009A000> 0x01 知识点 </font>

1. 预训练 pre-trainning
   - [huggingFace-FineWeb 数据集](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) --> 过滤后的最终文本数据 `44TB`
     - CommonCrawl 的网络爬虫
     - url-filtering : <https://dsi.ut-capitole.fr/blacklists/>
     - text-extraction : 去除 html 标记
     - lang-filtering : 保留 85% 以上英文的网页
     - ... 重复信息删除
     - PII remove : personally identifiable information remove 个人信息移除
     - 最终处理结果 : <https://huggingface.co/datasets/HuggingFaceFW/fineweb?row=0> -> Dataset Preview 部分
2. tokenization :
   - Converts text <---> sequences of symbols (/tokens) 尽可能的压缩文本数据量
     - why ?
   - <https://tiktokenizer.vercel.app/>
3. neural network training :
   1. 窗口上下文数量 maximum-context-length
   2. LLM Visualization : <https://bbycroft.net/llm>
   3. gpt :
      - 论文 : attention is all you need
      - TODO: 论文阅读工具...
      - GPT2 :
        - <https://github.com/openai/gpt-2.git>
        - 论文 : <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
          - [论文阅读：Language Models are Unsupervised Multitask Learners](https://zhuanlan.zhihu.com/p/711058430)
        - 复现项目 : `LLM.c` --> <https://github.com/karpathy/LLM.c/discussions/677>
      - llama-3 :
        - 论文 : [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)
   4. 网络结构 : `model.py`
4. (基础)模型 :
   - 分类
     1. base-model 基础模型
     2. instruct-model 指令模型
   - 本质上是对训练数据(互联网数据)的一种有损压缩
   - 训练数据是落后与真实世界的互联网的, 而模型的回答只是对历史互联网数据的模糊回忆, 因此其回答不可信
   - 当你复制 wikipedia 的内容来提问时, 模型可能输出和 wiki 完全一致的回答
     - 反刍 regurgitation : 直接引用已训练过的数据
     - 原因 : Wikipedia 被认为是高质量数据, 可能有较高的训练优先级(训练/看过 10 次, 以至于模型像人一样完全记住了)
   - 提示 prompt
     - in-context learning 上下文学习 --> 解决未知问题的一部分能力
     - (基础模型)的提示词 prompt 的撰写/优化
5. post-trainning 后训练 : supervised fine-tuning 监督微调 (`SFT`) --> 个性的出现
   - human labelers (人工标注者) 手动回答问题, 用于模型后训练
     - 数据集很小, 因此训练时间会很短
   - tokenization 的 encoding : 见上图的 `Conversation Protocol / Format`
     - NOTE: `<|im_start|>` 是一个标签, 基础模型并不包含, 其是新增的
     - 论文 : [Training language models to fol instructions with human feedback](https://arxiv.org/pdf/2203.02155)
     - 人工标注回答的指导文档通常有几十页之多
     - 开源的标注集 : [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)
     - 现在多使用 LLM 直接生成答案, 然后编辑它, 而不是从头手动编写, e.g. : [ultraChat](https://github.com/thunlp/UltraChat)
       - SFT 数据集 ?
       - 涵盖的方面 : <https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac>
       - 某些特定的技术/领域需要对应专家的参与标注确认
   - LLM psyhology 心理学
     - Hallucinations 幻觉 : 完全虚构信息
       - 消除方法 :
         1. > Use model interrogation to discover model's knowledge, and programmatically augment its training dataset with knowledge-based refusals in cases where the model doesn't know
            - 利用模型询问来发掘模型的知识，并在模型不知情的情况下，通过编程方式用基于知识的拒绝来扩充其训练数据集。
            - (用新的知识询问好的模型, 得出答案(推理), 然后将其加入训练集) : 网络中某个神经元值很高时, 让模型输出不知道(加强这种能力的训练)
         2. > Allow the model to search!
            - `<SEARCH_START>` 引入的新 token, 搜索引擎搜索, 爬虫, 进入上下文窗口(隐形), 后续的回答就进入了下一个 context-window (工作内存), 而该状态中包含了搜索的信息
       > - Knowledge in the parameters == Vague recollection (e.g. of something you read 1 month ago) (参数中的知识==模糊的回忆（例如1个月前读过的东西）)
       > - Knowledge in the tokens of the   context window == Working memory  (上下文窗口符号中的知识==工作记忆)
     - 模型的自我认知 Knowledge of self
     - Models need tokens to think
       - 模型下一个 tokens 的生成中计算量是有限的, 不可能一下子算出复杂的结果
       - 如果可以, 让模型使用工具(code), 而不是让模型将其全部内容都存储在 memory 中
     - 模型不擅长计数 : 数某个字符的个数
     - Models are not good with spelling --> tokenization 点副作用, 模型看到的并不是单词
       - tokenization 的根源是为了减少计算量, 直接输入字符将会是很大的计算量
         - 中文在这方面是还不是会更好 ?
     - Bunch of other small random stuff
6. reinforcement-learning 强化学习(`RL` 依旧属于 post-training 的范畴)
   - why ? --> 类似于人的上学 ?
   - 将问题多次询问模型, 得到其中正确的, 将其塞入作为未来的训练集中进行强化训练.
   - 新出现的阶段, 还不是很成熟
     - openAi 并未开源其方案
     - deepseek 的 R1 强化学习论文及其开源的重要性 --> 重新激发公众对使用 RL 训练 LLM 的兴趣
       - 论文 : [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
       - TODO: [论文解读DeepSeek-R1](https://zhuanlan.zhihu.com/p/19650946134)
       - thinking-model : 思考模型 FIXME:
         - **gpt-4O 是 SFT 模型, 基本没有进行 RL 训练(做了 RLHF)**
         - O1 的系列模型才是思考模型(用了 RL 训练)
   - 不可验证领域中的强化学习问题. E.g. : 幽默感, 读摘要, 诗歌创作, 创意写作
     - 论文 `RLHF` (Reinforcement Learning from Human Feedback) --> [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593)
       - 训练另一个完全独立奖励模型, 对被训练模型的结果进行评分(排序)
     - > We can run RL, in arbitrary domains! (even the unverifiable ones) This (empirically) improves the performance of the model, possibly due to the "discriminator - generator gap": In many cases, it is much easier to discriminate than to generate.  我们可以在任意域运行强化学习！（即使是无法验证的）这（经验上）提高了模型的性能，可能是由于“判别器-生成器差距”：在许多情况下，判别比生成容易得多。
     - RLHF 缺点 docnside :
       - > We are doing RL with respect to a lossy simulation of humans. It might be misleading!
       - > RL discovers ways to "game" the model. It discovers "adversarial examples" of the reward model.
         - 前期效果好 ? 后期有很多对抗性样本(adversarial examples), E.g. : the the the the 被认为是最好笑的笑话
         - 一般只运行几百次, 然后要剪枝
     - NOTE: 将小的专业的模型, 以 RL 的方式高效的塞到大的模型中 ?
     - 总结 : RLHF 并不完全等价于 RL, RLHF 不能永远运行,但是 RL 可以(其有个明确的判断标准)
7. future 未来的演化 :
   - multimodal (not just text but audio, images, video, natural conversations)
   - `tasks` -> `agents` (long, coherent, error-correcting contexts)
     - 工厂的自动化率 --> 数字世界中人工的监督
   - pervasive, invisible
   - computer-using
   - test-time training(实时学习)?, etc
     - context-window 资源的宝贵
   - [lmarena.ai](https://lmarena.ai/) : LLM 排行榜(人工)
   - `LMStudio` 本地运行模型
