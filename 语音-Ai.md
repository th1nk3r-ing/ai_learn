# <font color=#0099ff> **语音 & Ai** </font>

> `@think3r` 2026-04-06 19:51:05

## <font color=#009A000> Wisper.cpp </font>

> ref :
>
> 1. <https://github.com/ggml-org/whisper.cpp.git>
> 2. [长音频离线语音识别系统——基于Whisper.cpp的本地部署方案](https://zhuanlan.zhihu.com/p/1934266877867198244)
> 3. [FFmpeg whisper语音生成字幕](https://zhuanlan.zhihu.com/p/1939862458127880270)

### <font color=#FF4500> usage </font>

```sh
# 编译 :
cmake -B build
cmake --build build --config Release

# 模型下载 :
bash ./models/download-ggml-model.sh large-v3  # 自带的模型
curl -L -O https://huggingface.co/second-state/whisper-large-zh-cv11-GGML/resolve/main/ggml-whisper-large-zh-cv11.bin # hugging face 上的中文优化版本模型

# 使用
./build/bin/whisper-cli --model models/ggml-large-v3.bin --file ~/Movies/youtube/testSub/input.wav  --print-colors -osrt   # 输出 srt 和 wav 同级目录
./build/bin/whisper-cli --model models/ggml-whisper-large-zh-cv11.bin --file ~/Movies/youtube/testSub/input.wav  --print-colors -l zh -osrt  # 输出的颜色为置信度 (中文好像暂时有乱码)
```

NOTE: the whisper-cli example currently runs only with **16-bit WAV** files (暂时仅支持 16bit wav 输入), so make sure to convert your input before running the tool. For example, you can use ffmpeg like this: `ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav`

### <font color=#FF4500> ~~coreML 加速~~ </font>

```sh
# mac 下的 core-ml 加速
python3 -m venv $(pwd)/build && source bin/activate # python3 环境创建
pip install ane_transformers openai-whisper coremltools # 依赖安装
cmake -B build -DWHISPER_COREML=1
cmake --build build -j --config Release

./models/generate-coreml-model.sh models/ggml-large-v3.bin
```

### <font color=#FF4500> 模型对比 </font>

Whisper 提供多个模型版本，主要差异在于参数数量、所需内存、处理速度和识别准确率。以下是各模型的简要对比：

| 模型     | 参数量   | ggml体积 | 典型内存   | 特点   |
| ------ | ----- | ------ | ------ | -------- |
| tiny   | 39M   | ~75MB  | ~300MB | 极快，精度低    |
| base   | 74M   | ~142MB | ~400MB | 入门平衡       |
| small  | 244M  | ~466MB | ~850MB | **推荐默认**   |
| medium | 769M  | ~1.5GB | ~2GB   | 高精度         |
| large  | 1.55B | ~2.9GB | ~4GB   | 最强精度     |

- 模型越大 → 准确率↑，速度↓，内存↑
- English-only 为 `.en` 结尾, 默认的为多语言版本
- large 系列的版本演进
  1. large-v1 : 初始版本, 已基本淘汰
  2. large-v2 : 更多训练轮次 + 正则化, 更稳定
  3. large-v3（当前主流）: 更大训练数据（百万小时级）, Mel bins: 80 → 128（更细频谱）, 精度最高
     - 👉 默认选它：large-v3
- turbo: 基于 large 的蒸馏, 速度有所提升(某些情况下会有精度问题)
- 使用 :
  - whisper的中文训练数据里不知道混进去多少来自字幕组的奇怪句子... 经常在结尾和开头混入

## <font color=#009A000> TODO: 阿里 funasr 的 paraformer </font>

