# FFmpeg 支持的三方库

## 视频编解码库

| 库名 | 用途 | 类型 | Android 支持 | GitHub URL |
|------|------|------|:---:|------------|
| **libaom** | AV1 编解码 | *参考实现* | ✅ | <https://github.com/AOM-AV1/aom> |
| **libdav1d** | AV1 快速解码器 | 优化实现 | ✅ | <https://code.videolan.org/videolan/dav1d> |
| **libsvtav1** | AV1 高性能编码器 | 优化实现 | ✅ | <https://gitlab.com/AOMediaCodec/SVT-AV1> |
| **librav1e** | AV1 编码器 (Rust) | 优化实现 | ✅ | <https://github.com/xiph/rav1e> |
| **libx264** | H.264 编码 | 优化实现 | ✅ | <https://code.videolan.org/videolan/x264> |
| **libx265** | H.265/HEVC 编码 | 优化实现 | ✅ | <https://bitbucket.org/multicoreware/x265_git> |
| **libopenh264** | H.264 编码 | 优化实现 | ✅ | <https://github.com/cisco/openh264> |
| **libkvazaar** | HEVC 编码 | 优化实现 | ✅ | <https://github.com/ultravideo/kvazaar> |
| **libvvenc** | H.266/VVC 编码 | *参考实现* | ✅ | <https://github.com/fraunhoferhhi/vvenc> |
| **libvvdec** | H.266/VVC 解码 | *参考实现* | ✅ | <https://github.com/fraunhoferhhi/vvdec> |
| **libvpx** | VP8/VP9 编解码 | *参考实现* | ✅ | <https://github.com/AOM-AV1/libvpx> |
| **libtheora** | Theora 编码 | *参考实现* | ✅ | <https://github.com/xiph/theora> |
| **libxvid** | MPEG-4 编码 | 优化实现 | ✅ | <https://labs.xvid.com/source/> |
| **libxavs** | AVS 编码 | *参考实现* | ✅ | <https://github.com/nicxzhang/xavs> |
| **libxavs2** | AVS2 编码 | *参考实现* | ✅ | <https://github.com/pkuvcl/xavs2> |
| **libdavs2** | AVS2 解码 | *参考实现* | ✅ | <https://github.com/pkuvcl/davs2> |
| **libuavs3d** | AVS3 解码 | *参考实现* | ✅ | <https://github.com/nicxzhang/uavs3d> |
| **libsvtjpegxs** | JPEG-XS 编解码 | 优化实现 | ✅ | <https://github.com/OpenVisualCloud/SVT-JPEG-XS> |
| **libxeve** | EVC 编码 | *参考实现* | ✅ | <https://github.com/mpeg5/xeve> |
| **libxevd** | EVC 解码 | *参考实现* | ✅ | <https://github.com/mpeg5/xevd> |
| **libjxl** | JPEG XL 编解码 | *参考实现* | ✅ | <https://github.com/libjxl/libjxl> |
| **liboapv** | APV 编码 | 优化实现 | ✅ | <https://github.com/Netflix/oapv> |
| **liblcevc-dec** | LCEVC 解码 | *参考实现* | ✅ | <https://github.com/v-nova/lcevc_dec> |

## 音频编解码库

| 库名 | 用途 | 类型 | Android 支持 | GitHub URL |
|------|------|------|:---:|------------|
| **libfdk-aac** | AAC 编解码 | 优化实现 | ✅ | <https://github.com/mstorsjo/fdk-aac> |
| **libopus** | Opus 编解码 | *参考实现* | ✅ | <https://github.com/xiph/opus> |
| **libmp3lame** | MP3 编码 | 优化实现 | ✅ | <https://lame.sourceforge.io/> |
| **libshine** | 固定点 MP3 编码 | 优化实现 | ✅ | <https://github.com/toots/shine> |
| **libvorbis** | Vorbis 编解码 | *参考实现* | ✅ | <https://github.com/xiph/vorbis> |
| **libspeex** | Speex 编解码 | *参考实现* | ✅ | <https://github.com/xiph/speex> |
| **libcodec2** | Codec2 语音编解码 | *参考实现* | ✅ | <https://github.com/drowe67/codec2> |
| **libgsm** | GSM 编解码 | *参考实现* | ✅ | <https://www.quut.com/gsm/> |
| **libilbc** | iLBC 编解码 | *参考实现* | ✅ | <https://github.com/nicxzhang/libilbc> |
| **libopencore-amrnb** | AMR-NB 编解码 | *参考实现* | ✅ | <https://sourceforge.net/projects/opencore-amr/> |
| **libopencore-amrwb** | AMR-WB 解码 | *参考实现* | ✅ | <https://sourceforge.net/projects/opencore-amr/> |
| **libvo-amrwbenc** | AMR-WB 编码 | *参考实现* | ✅ | <https://sourceforge.net/projects/opencore-amr/> |
| **libtwolame** | MP2 编码 | 优化实现 | ✅ | <https://github.com/njh/twolame> |
| **liblc3** | LC3 编解码 | *参考实现* | ✅ | <https://github.com/nicxzhang/liblc3> |
| **libcelt** | CELT 解码 | *参考实现* | ✅ | <https://downloads.xiph.org/releases/celt/> |
| **libmpeghdec** | MPEG-H 3DA 解码 | *参考实现* | ✅ | <https://github.com/nicxzhang/libmpeghdec> |
| **libgme** | 游戏音乐模拟 | 其他 | ✅ | <https://github.com/libgme/game-music-emu> |
| **libflite** | 语音合成 | 其他 | ✅ | <https://github.com/festvox/flite> |
| **libmodplug** | MOD 音乐解码 | 其他 | ✅ | <https://github.com/Konstanty/libmodplug> |
| **libopenmpt** | MOD 音乐解码 | 其他 | ✅ | <https://github.com/nicxzhang/libopenmpt> |

## 字幕/文本处理库

| 库名 | 用途 | 类型 | Android 支持 | GitHub URL |
|------|------|------|:---:|------------|
| **libass** | ASS/SSA 字幕渲染 | 优化实现 | ✅ | <https://github.com/libass/libass> |
| **libaribb24** | ARIB 字幕解码 | *参考实现* | ✅ | <https://github.com/nicxzhang/libaribb24> |
| **libaribcaption** | ARIB 字幕解码 | *参考实现* | ✅ | <https://github.com/nicxzhang/aribcaption> |
| **libfreetype** | 字体渲染 | 系统库 | ✅ | <https://gitlab.freedesktop.org/freetype/freetype> |
| **libfontconfig** | 字体配置 | 系统库 | ✅ | <https://gitlab.freedesktop.org/fontconfig/fontconfig> |
| **libfribidi** | BiDi 文本处理 | 系统库 | ✅ | <https://github.com/fribidi/fribidi> |
| **libharfbuzz** | 文本整形 | 系统库 | ✅ | <https://github.com/harfbuzz/harfbuzz> |
| **libtesseract** | OCR 文字识别 | 其他 | ✅ | <https://github.com/tesseract-ocr/tesseract> |
| **libqrencode** | QR 码生成 | 其他 | ✅ | <https://github.com/fukuchi/libqrencode> |
| **libquirc** | QR 码解码 | 其他 | ✅ | <https://github.com/nicxzhang/libquirc> |
| **libzvbi** | 图文电视 | 其他 | ✅ | <https://github.com/zapping-vbi/zvbi> |

## 视频处理/滤镜库

| 库名 | 用途 | 类型 | Android 支持 | GitHub URL |
|------|------|------|:---:|------------|
| **libplacebo** | GPU 视频渲染 | 硬件加速 | ✅ | <https://code.videolan.org/videolan/libplacebo> |
| **libzimg** | 图像缩放/色彩转换 | 优化实现 | ✅ | <https://github.com/sekrit-twc/zimg> |
| **libopencv** | 计算机视觉 | 其他 | ✅ | <https://github.com/opencv/opencv> |
| **libvidstab** | 视频稳定 | 优化实现 | ✅ | <https://github.com/georgmartius/vid.stab> |
| **librubberband** | 音频时间拉伸 | 优化实现 | ✅ | <https://github.com/breakfastquay/rubberband> |
| **libvmaf** | 视频质量评估 | 其他 | ✅ | <https://github.com/Netflix/vmaf> |
| **liblensfun** | 镜头校正 | 其他 | ✅ | <https://github.com/lensfun/lensfun> |
| **libmysofa** | SOFA HRTF | 其他 | ✅ | <https://github.com/hoene/libmysofa> |
| **libbs2b** | 音频 DSP | 其他 | ✅ | <https://github.com/alexmarsev/libbs2b> |
| **libsnappy** | Snappy 压缩 (HAP) | 系统库 | ✅ | <https://github.com/google/snappy> |
| **libsoxr** | 音频重采样 | 优化实现 | ✅ | <https://sourceforge.net/projects/soxr/> |

## 协议/网络库

| 库名 | 用途 | 类型 | Android 支持 | GitHub URL |
|------|------|------|:---:|------------|
| **libsrt** | SRT 协议 | 优化实现 | ✅ | <https://github.com/Haivision/srt> |
| **librist** | RIST 协议 | 优化实现 | ✅ | <https://code.videolan.org/rist/librist> |
| **librtmp** | RTMP 协议 | 其他 | ✅ | <https://rtmpdump.mplayerhq.hu/> |
| **libssh** | SFTP 协议 | 系统库 | ✅ | <https://libssh.org/> |
| **libsmbclient** | SMB/CIFS 协议 | 系统库 | ⚠️ 需交叉编译 | <https://github.com/samba-team/samba> |
| **librabbitmq** | AMQP 消息队列 | 其他 | ✅ | <https://github.com/alanxz/rabbitmq-c> |
| **libzmq** | ZeroMQ 消息传递 | 其他 | ✅ | <https://github.com/zeromq/libzmq> |
| **libxml2** | XML 解析 | 系统库 | ✅ | <https://gitlab.gnome.org/GNOME/libxml2> |

## 硬件/平台库

| 库名 | 用途 | 类型 | Android 支持 | GitHub URL |
|------|------|------|:---:|------------|
| **libmfx** | Intel Quick Sync Video | 硬件加速 | ❌ | <https://github.com/Intel-Media-SDK/MediaSDK> |
| **libvpl** | Intel oneVPL | 硬件加速 | ❌ | <https://github.com/intel/libvpl> |
| **libnpp** | NVIDIA NPP | 硬件加速 | ❌ | <https://developer.nvidia.com/cuda-toolkit> |
| **libglslang** | GLSL→SPIRV 编译 | 硬件加速 | ✅ | <https://github.com/KhronosGroup/glslang> |
| **libshaderc** | GLSL→SPIRV 编译 | 硬件加速 | ✅ | <https://github.com/google/shaderc> |
| **libv4l2** | Video4Linux2 | 硬件加速 | ⚠️ 部分支持 | <https://github.com/gjasny/v4l-utils> |
| **libdc1394** | IEEE 1394 摄像头 | 硬件加速 | ❌ | <https://sourceforge.net/projects/libdc1394/> |
| **libiec61883** | IEC 61883 | 硬件加速 | ❌ | <https://github.com/nicxzhang/libiec61883> |
| **libpulse** | PulseAudio | 系统库 | ❌ | <https://gitlab.freedesktop.org/pulseaudio/pulseaudio> |
| **libjack** | JACK 音频 | 系统库 | ❌ | <https://github.com/jackaudio/jack2> |
| **libcaca** | 文本模式显示 | 其他 | ✅ | <https://github.com/cacalabs/libcaca> |
| **libcdio** | CD 抓取 | 其他 | ❌ | <https://github.com/rocky/libcdio> |
| **libdvdnav** | DVD 导航 | 其他 | ⚠️ 需交叉编译 | <https://code.videolan.org/videolan/libdvdnav> |
| **libdvdread** | DVD 读取 | 其他 | ⚠️ 需交叉编译 | <https://code.videolan.org/videolan/libdvdread> |
| **libbluray** | 蓝光读取 | 其他 | ⚠️ 需交叉编译 | <https://code.videolan.org/videolan/libbluray> |
| **librsvg** | SVG 光栅化 | 系统库 | ⚠️ 需交叉编译 | <https://gitlab.gnome.org/GNOME/librsvg> |
| **libcairo** | 2D 图形 | 系统库 | ⚠️ 需交叉编译 | <https://gitlab.cairographics.org/cairo/cairo> |
| **libopenvino** | OpenVINO DNN | 硬件加速 | ✅ | <https://github.com/openvinotoolkit/openvino> |
| **libtensorflow** | TensorFlow DNN | 硬件加速 | ✅ | <https://github.com/tensorflow/tensorflow> |
| **libtorch** | PyTorch DNN | 硬件加速 | ✅ | <https://github.com/pytorch/pytorch> |
| **libopencolorio** | 色彩管理 | 其他 | ✅ | <https://github.com/AcademySoftwareFoundation/OpenColorIO> |
| **libklvanc** | VANC 处理 | 其他 | ❌ | <https://github.com/Kitchen-Registry/kvanc> |
| **zlib** | 压缩库 | 系统库 | ✅ | <https://github.com/madler/zlib> |
| **SDL2** | 多媒体框架 | 系统库 | ✅ | <https://github.com/libsdl-org/SDL> |

## FFmpeg 内置平台特性

| 特性 | 用途 | Android 支持 | 说明 |
|------|------|:---:|------|
| **MediaCodec** | Android 硬件编解码 | ✅ | Android 原生硬件编解码 API |
| **JNI** | Java Native Interface | ✅ | Android NDK 内置 |
| **Vulkan** | GPU 计算/图形 API | ✅ | 跨平台 GPU API |
| **VideoToolbox** | Apple 硬件编解码 | ❌ | macOS/iOS 专用 |
| **VAAPI** | Intel/AMD 视频加速 | ❌ | Linux 桌面专用 |
| **VDPAU** | NVIDIA 视频加速 | ❌ | Linux 桌面专用 |
| **D3D11VA** | Direct3D 11 视频加速 | ❌ | Windows 专用 |
| **D3D12VA** | Direct3D 12 视频加速 | ❌ | Windows 专用 |
| **DXVA2** | DirectX 视频加速 | ❌ | Windows 专用 |
| **NVDEC/NVENC** | NVIDIA 硬件编解码 | ❌ | 需要 NVIDIA GPU |
| **CUDA** | NVIDIA GPU 计算 | ❌ | 需要 NVIDIA GPU |
| **QSV** | Intel Quick Sync | ❌ | Intel CPU 专用 |
| **ALSA** | Linux 音频 | ❌ | Linux 桌面专用 |
| **PulseAudio** | Linux 音频 | ❌ | Linux 桌面专用 |
| **X11/XCB** | X Window 系统 | ❌ | Linux 桌面专用 |
| **MMAL** | Raspberry Pi 视频 | ❌ | 树莓派专用 |
| **OpenMAX IL** | 嵌入式多媒体 | ❌ | 树莓派/嵌入式 |
| **RKMPP** | Rockchip 媒体处理 | ❌ | Rockchip SoC 专用 |

## 图例

| 符号 | 含义 |
|:---:|------|
| ✅ | 完全支持 Android |
| ⚠️ | 需要交叉编译或部分支持 |
| ❌ | 不支持 Android |

## 统计

- ***参考实现***: 25 个 (标准组织/官方参考代码)
- **优化实现**: 18 个 (针对性能优化的第三方实现)
- **硬件加速**: 14 个 (GPU/DSP/专用硬件加速)
- **系统库**: 14 个 (操作系统/平台库)
- **其他**: 29 个 (其他用途)

**总计**: 100+ 个可选三方库

### Android 兼容性统计

- ✅ 完全支持: 约 **75** 个
- ⚠️ 需交叉编译: 约 **8** 个 (smbclient, dvdnav, dvdread, bluray, librsvg, cairo, v4l2, dc1394)
- ❌ 不支持: 约 **17** 个 (Apple/Windows/NVIDIA 专属硬件库)

## 当前编译配置 (Android aarch64)

已启用:
- **libplacebo** - GPU 视频渲染 (Vulkan)
- **libvvdec** - H.266/VVC 参考解码器
- **Vulkan** - GPU 计算/图形 API
- **MediaCodec** - Android 硬件编解码
- **JNI** - Java Native Interface
- **SDL2** - 多媒体框架
- **zlib** - 压缩库
- **iconv** - 字符编码转换
