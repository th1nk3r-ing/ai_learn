# GitHub 开源软件清单

## gh 查看 release 说明

使用 GitHub CLI (`gh`) 查看仓库的 release 发布情况，需先 `gh auth login` 认证。

```bash
# 列出仓库最近的 release
gh release list -R <owner/repo> --limit 10

# 查看指定 tag 的 release 详情（发布说明、assets 等）
gh release view <tag> -R <owner/repo>

# 下载指定 release 的 assets
gh release download <tag> -R <owner/repo>

# 部分仓库不使用 GitHub Releases，改用 tags 查看版本
gh api repos/<owner/repo>/tags -q '.[].name'
```

> **注意**：并非所有仓库都有 GitHub Releases 页面，有些项目（如 FFmpeg）仅使用 tags 标记版本，`gh release list` 会返回空。

---

## 软件清单

| 本地文件/文件夹 | 平台 | GitHub URL | 本地版本 | 查本地版本方法 | gh 最新版本 | 是否最新 |
|----------------|------|-----------|---------|--------------|------------|------|
| ffmpeg | Win+Linux | <https://github.com/BtbN/FFmpeg-Builds> | n8.1.1 | `ffmpeg -version` | latest (2026-06-08) | ❌ |
| RenderDoc_1.44_64 | Win | <https://github.com/baldurk/renderdoc> | v1.44 | `exiftool renderdocui.exe \| grep "Product Version"` | v1.44 | ✅ |
| scrcpy | Win+Linux(软链接) | <https://github.com/Genymobile/scrcpy> | 4.0 | `scrcpy --version` | v4.0 | ✅ |
| jadx-gui-1.5.5 | Win+Linux(软链接) | <https://github.com/skylot/jadx> | 1.5.5 | 文件夹名推断 | v1.5.5 | ✅ |
| draw.io | Win | <https://github.com/jgraph/drawio-desktop> | 30.0.4 | `exiftool draw.io.exe \| grep "Product Version"` | v30.0.4 | ✅ |
| mpv-x86_64-v3 | Win+Linux(软链接) | <https://github.com/zhongfly/mpv-winbuild> | v0.41.0 | `mpv --version` | 2026-06-08 | ❌ |
| YUView-Win | Win | <https://github.com/IENT/YUView> | v2.14-322 | `strings YUView.exe \| grep` | v2.14 | ✅ |
| MediaInfo_GUI | Win | <https://github.com/MediaArea/MediaInfo> | 26.01.0 | `exiftool MediaInfo.exe \| grep "Product Version"` | v26.05 | ❌ |
| ImageGlass_x64 | Win | <https://github.com/d2phap/ImageGlass> | 9.4.1.15 | `exiftool ImageGlass.exe \| grep "Product Version"` | 9.5.0.515 | ❌ |
| WinDirStat.exe | Win | <https://github.com/windirstat/windirstat> | 2.6.2 | `exiftool WinDirStat.exe \| grep "Product Version"` | v2.6.2 | ✅ |
| ContextMenuManager.NET.4.0.exe | Win | <https://github.com/BluePointLilac/ContextMenuManager> | 3.3.3.1 | `exiftool ContextMenuManager.NET.4.0.exe \| grep "Product Version"` | 3.3.3.1 | ✅ |
| Snipaste-2.10.6-x64 | Win | <https://github.com/Snipaste/feedback> | 2.11.2 | `exiftool Snipaste.exe \| grep "Product Version"` | 无 release | - |
| yt-dlp | Linux | <https://github.com/yt-dlp/yt-dlp> | 2026.03.17 | `yt-dlp --version` | 2026.03.17 | ✅ |
| glslang | Linux | <https://github.com/KhronosGroup/glslang> | 16.3.0 | `glslang --version` | 16.3.0 | ✅ |
| bloaty | Linux | <https://github.com/google/bloaty> | 1.1 | `bloaty --version` | v1.1 | ✅ |
| tldr | Linux | <https://github.com/tldr-pages/tlrc> | v1.11.1 | `tldr --version` | v1.13.1 | ❌ |
| axel-2.17.14 | Linux | <https://github.com/axel-download-accelerator/axel> | 2.17.14 | `axel --version` | v2.17.14 | ✅ |

共 17 个条目，16 个 GitHub 仓库。
