# MicroCode

本地编码 Agent。模型每一步要么直接回答，要么调用工具，在当前仓库里读代码、改文件、跑命令。

运行时只依赖 Python 标准库。向量记忆是可选安装。

## 环境

- Python 3.11+
- 一个兼容 Anthropic Messages 或 OpenAI Chat Completions 的接口

## 安装

```powershell
pip install -e ".[dev]"
```

要启用本地向量记忆（`all-MiniLM-L6-v2`）：

```powershell
pip install -e ".[embeddings]"
```

没装这个额外依赖时，记忆检索只用 BM25 和 TF-IDF。

## 配置

```powershell
copy .env.example .env
```

在 `.env` 里填写：

| 变量 | 作用 |
|---|---|
| `MICROCODE_API_KEY` | 接口密钥。`.env` 已被 git 忽略 |
| `MICROCODE_BASE_URL` | 接口地址。地址里含 `anthropic` 时走 Anthropic，否则走 OpenAI |
| `MICROCODE_MODEL` | 模型名 |

不配密钥时可以用 `--mock`，离线走脚本模型，不访问网络。

## 运行

```powershell
microcode
microcode "给这个仓库补一个测试"
microcode --once "列出 microcode 目录"
microcode --mock
microcode --readiness
```

交互终端默认是全屏 TUI。`--once`、管道，以及非 TTY 会退回一行一行的 REPL。`--tui` 强制全屏，`--no-tui` 强制行式。

常用参数：

| 参数 | 作用 |
|---|---|
| `--cwd` | 工作区根目录，默认当前目录 |
| `--yes` | 写文件和跑命令不再询问 |
| `--resume [id]` | 恢复最近一次会话，或指定会话 id |
| `--readiness` | 只检查本地配置，不调用模型 |

会话里输入 `/help` 查看斜杠命令。

## 能力

**Agent Loop。** 同一轮里可以连续读文件、检索、改写和跑命令。`explore` / `plan` 是只读 sub-agent，`general` 可以改工作区。子循环结束后，父对话只收到最后一段摘要。

**工具。** 列目录、读文件、搜索、符号跳转、改文件、git、跑命令、跑测试、网页抓取和搜索。写文件和跑命令默认要你批准，路径限制在工作区内。批准前可以改掉模型给出的正文再落盘，之后可以用 `/undo`、`/redo`、`/rewind` 回退。

**记忆。** 笔记按 `user` / `project` / `local` 分范围，并按 `working` / `short-term` / `long-term` / `archival` 分层。检索把 MiniLM 向量、BM25 和 TF-IDF 做 RRF 融合，再交给模型 Rerank，只把相关笔记注入 system prompt。新开会话不读旧消息。

**记录。** 每轮步骤追加到 `.microcode/turns.jsonl`，不进入模型上下文。`/fixture` 把当前 session 导出成离线 pytest，重放 tool call 和最终回答。

工作区状态写在 `.microcode/` 下，这个目录已被 git 忽略。

## 测试

```powershell
pytest
```

回合磁带和向量模型在测试里默认关闭，不会写当前仓库的 `.microcode/`，也不会下载 MiniLM。
