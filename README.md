# prompt-assembler
Prompt Assembler - 智能提示词（Prompt）生成引擎


**Prompt Assembler** 是一个强大而灵活的命令行工具，旨在将编写复杂的大语言模型（LLM）提示词（Prompt）从繁琐的手工"复制粘贴"中解放出来。它利用 Jinja2 模板引擎，允许你通过编程的方式，动态地、智能地从你的项目文件和目录中组装上下文，生成最终的提示词。

> 告别手动维护巨大的 `prompt.txt` 文件，让你的 Prompt 和你的代码库同步进化！

✨ 功能特性
------

-   **模板驱动**：使用强大的 [Jinja2](https://jinja.palletsprojects.com/) 模板语言定义你的 Prompt 结构。

-   **文件嵌入** (`include_file`): 精确地将单个文件内容嵌入到提示词中。

-   **目录扫描** (`include_dir`): 递归地包含整个目录的文件，并支持强大的 `glob` 模式和关键词过滤。

-   **模糊匹配** (`include_fuzzy`): 根据不精确的文件名智能查找并包含文件，告别因拼写错误或路径遗忘带来的烦恼。

-   **Token 报告**: 在生成 Prompt 后，自动提供详细的 Token 消耗报告，帮助你管理和优化上下文窗口。

-   **跨平台剪贴板**: 生成的 Prompt 会自动复制到你的系统剪贴板中，方便立即使用。

-   **配置灵活**: 支持自定义输出文件、选择 Token 计算模型等。

📦 安装指南
-------

1.  **克隆项目**

    Bash

    ```
    git clone https://github.com/your-username/prompt-assembler.git
    cd prompt-assembler

    ```

2.  **安装依赖**

    建议在 Python 虚拟环境中安装依赖。

    Bash

    ```
    # 创建虚拟环境 (可选但推荐)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # 安装依赖
    pip install -r requirements.txt

    ```

3. **运行示例**

    Bash

    ```
    python3 prompt_assembler.py example/my_prompt.j2
    ```


📚 使用教程
-------

使用 Prompt Assembler 只需三步：创建目录结构 -> 编写模板 -> 运行命令。

### 步骤 1: 准备你的项目目录结构

假设我们有一个简单的项目，结构如下：

```
my_project/
├── src/
│   ├── main.py
│   ├── utils/
│   │   ├── helpers.py
│   │   └── data_processor.py
│   └── config.json
├── docs/
│   └── requirements_v2.md
├── templates/
│   └── my_prompt.j2
└── prompt_assembler.py

```

### 步骤 2: 编写你的 Jinja2 模板

在 `templates/` 目录下，创建一个 `.j2` 文件，例如 `my_prompt.j2`。在这个模板里，你将定义 Prompt 的最终结构，并使用我们提供的自定义函数来"拉取"上下文。

**`templates/my_prompt.j2`**

Django

```
你是一位资深的 Python 架构师。我需要你为以下项目撰写一份技术架构评审。

这是项目的核心需求文档：
{{ include_file('../docs/requirements_v2.md') }}

这是项目的主要源代码。请注意，我排除了一些不必要的文件：
{{ include_dir(
    '../src',
    glob_pattern='**/*.py',
    exclude_patterns=['**/__pycache__/*']
) }}

我不太确定配置文件的确切名称，但它应该在 src 目录下，并且和 'config' 相关：
{{ include_fuzzy('../src/configuration.json') }}

基于以上所有信息，请从以下几个方面进行评审：
1.  代码结构是否合理？
2.  是否存在明显的性能瓶颈？
3.  有没有可以改进的设计模式？

```

### 步骤 3: 运行命令生成 Prompt

打开终端，运行 `prompt_assembler.py` 脚本，并指向你的模板文件。

Bash

```
python prompt_assembler.py templates/my_prompt.j2 -o final_prompt.txt

```

### 步骤 4: 查看结果

运行后，你会得到：

1.  **终端输出**:

    ```
    ✅ 结果已经复制到剪贴板
    ✅ 结果已经保存到 ./final_prompt.txt
    INFO: 模糊查找 '../src/configuration.json' -> 匹配到 'src/config.json' (Partial Ratio 评分: 90.00%)

    --- Token Usage Report (Top 10) ---
       Tokens |   Percentage | File Path
    ---------+--------------+-----------------------
        1,205 |       45.15% | src/utils/helpers.py
          850 |       31.84% | src/main.py
          412 |       15.43% | docs/requirements_v2.md
          180 |        6.74% | src/utils/data_processor.py
           22 |        0.82% | src/config.json

    ✅ Total Tokens: 2,669

    ```

2.  **文件 `final_prompt.txt`**: 一个包含了所有文件内容的、可以直接使用的完整 Prompt。

3.  **剪贴板**: 完整 Prompt 已在你的剪贴板里，随时可以粘贴到 LLM 对话框中。

🛠️ 函数 API 参考
-------------

你可以在 Jinja2 模板中使用以下三个强大的函数：

### 1\. `include_file()`

精确嵌入单个文件的内容。

-   **签名**: `include_file(path: str, show_filename: bool = True, tag: str = "PROMPT_SNIPPET_FILE")`

-   **参数**:

    -   `path`: 必填，相对于模板文件的文件路径。

    -   `show_filename`: 可选，是否在包裹内容的 XML 标签中显示 `path` 属性。

    -   `tag`: 可选，用于包裹内容的 XML 标签名。设为 `'none'` 则不添加任何标签。

### 2\. `include_dir()`

递归地嵌入一个目录下的多个文件，并提供强大的过滤能力。

-   **签名**: `include_dir(path: str, glob_pattern: str = "**/*", exclude_patterns: list = None, exclude_keywords: list = None, tag: str = "directory")`

-   **参数**:

    -   `path`: 必填，相对于模板文件的目录路径。

    -   `glob_pattern`: 可选，用于匹配文件的 glob 模式（例如 `**/*.py`）。

    -   `exclude_patterns`: 可选，一个 `glob` 模式列表，匹配到的文件将被排除。

    -   `exclude_keywords`: 可选，一个字符串列表，如果文件内容包含任一关键词，该文件将被排除。

    -   `tag`: 可选，用于包裹整个目录内容的 XML 标签名。

### 3\. `include_fuzzy()`

根据近似路径模糊查找并嵌入单个文件。

-   **签名**: `include_fuzzy(approximate_path: str, similarity_threshold: int = 85, tag: str = "PROMPT_SNIPPET_FILE")`

-   **参数**:

    -   `approximate_path`: 必填，一个可能不准确的文件路径。脚本将在其父目录中进行模糊搜索。

    -   `similarity_threshold`: 可选，匹配的相似度分数阈值（0-100）。

    -   `tag`: 可选，同 `include_file`。

⚙️ 命令行选项
--------

```
usage: prompt_assembler.py [-h] [-o OUTPUT] [--no-clipboard] [--model MODEL] template_file

一个智能的提示词（Prompt）生成引擎。

positional arguments:
  template_file         Jinja2 模板文件的路径 (.j2)。

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        保存结果的输出文件路径 (默认: output.txt)。
  --no-clipboard        不将结果复制到系统剪贴板。
  --model MODEL         用于 token 计算的模型 (默认: gpt-4o)。

```

🤝 贡献
-----

欢迎提交 Pull Requests 或开启 Issues 来改进这个项目！

📄 许可证
------

本项目采用 [MIT 许可证](https://www.google.com/search?q=LICENSE&authuser=2)。
