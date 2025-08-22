from pathlib import Path
from functools import partial
import sys
import jinja2
from rapidfuzz import process, fuzz
import pyperclip
import tiktoken
import argparse
from typing import List, Optional, Set
import re

# --- 常量定义 ---
DEFAULT_MODEL = "gpt-4o"
DEFAULT_FILE_TAG = "PROMPT_SNIPPET_FILE"
DEFAULT_DIR_TAG = "PROMPT_SNIPPET_DIRECTORY"


def remove_pattern(input_string: str) -> str:
    """
    从字符串中移除所有符合连续注释之间的空白
    """
    pattern = r"\#}\s+\{\#"
    return re.sub(pattern, "", input_string)


class PromptAssembler:
    """
    一个智能的提示词（Prompt）生成引擎。
    它使用 Jinja2 模板，并注入了强大的自定义函数，
    用于从文件和目录中智能地组合上下文。
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        初始化 PromptAssembler。
        :param model: 用于计算 token 的模型名称。
        """
        self.env: Optional[jinja2.Environment] = None
        self.base_dir: Optional[Path] = None
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.token_stats: dict[str, int] = {}
        self.included_paths: Set[Path] = set()

    def _calc_token(self, text: str) -> int:
        """计算给定字符串的 token 数量。"""
        return len(self.tokenizer.encode(text))

    def _initialize_environment(self, template_path: Path):
        """根据模板路径创建并配置 Jinja2 环境。"""
        self.base_dir = template_path.parent
        loader = jinja2.FileSystemLoader(self.base_dir)
        self.env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.token_stats = {}
        self.included_paths = set()
        self.env.globals["include_file"] = partial(self._include_file)
        self.env.globals["include_fuzzy"] = partial(self._include_fuzzy)
        self.env.globals["include_dir"] = partial(self._include_dir)

    def _resolve_path(self, path_str: str) -> Path:
        """将模板中的相对路径解析为相对于模板文件本身的绝对路径。"""
        if self.base_dir is None:
            raise RuntimeError(
                "Environment has not been initialized. Call render() first."
            )
        return (self.base_dir / Path(path_str)).resolve()

    def _include_file(
        self,
        path: str,
        base_dir: Optional[str] = None,
        show_filename: bool = True,
        tag: str = DEFAULT_FILE_TAG,
    ) -> str:
        """
        嵌入单个文件内容。
        """
        if base_dir:
            file_path = Path(base_dir).resolve() / Path(path)
        else:
            file_path = self._resolve_path(path)
        if file_path in self.included_paths:
            print(
                f"WARNING: The file '{path}' has been included more than once.",
                file=sys.stderr,
            )
        self.included_paths.add(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"文件未找到: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise IOError(f"读取文件 '{file_path}' 时出错: {e}") from e
        token_count = self._calc_token(content)
        self.token_stats[path] = self.token_stats.get(path, 0) + token_count
        if tag.lower() == "none":
            return content
        path_attr = f' path="{path}"' if show_filename else ""
        return f"\n<{tag}{path_attr}>\n{content}\n</{tag}>\n"

    def _include_fuzzy(
        self,
        approximate_path: str,
        similarity_threshold: int = 85,
        tag: str = DEFAULT_FILE_TAG,
    ) -> str:
        """
        根据近似路径模糊查找并嵌入文件。
        如果找到多个100分的完美匹配，则会报错，强制用户明确指定。
        """
        target_path = Path(approximate_path)
        search_dir = self._resolve_path(str(target_path.parent))
        target_filename = target_path.name
        if not search_dir.is_dir():
            raise FileNotFoundError(f"用于模糊查找的目录不存在: {search_dir}")
        choices = [item.name for item in search_dir.iterdir() if item.is_file()]
        if not choices:
            raise FileNotFoundError(
                f"在目录 '{search_dir}' 中没有找到任何文件可供匹配。"
            )
        perfect_matches = process.extract(
            target_filename,
            choices,
            scorer=fuzz.partial_ratio,
            score_cutoff=100,
        )
        if len(perfect_matches) > 1:
            conflicting_files = [match[0] for match in perfect_matches]
            raise ValueError(
                f"模糊查找发现歧义: 查询 '{target_filename}' 存在多个完美匹配项。\n"
                f"请提供更具体的文件路径以消除歧义。\n"
                f"冲突文件列表: {conflicting_files}"
            )
        result = process.extractOne(
            target_filename,
            choices,
            scorer=fuzz.partial_ratio,
            score_cutoff=similarity_threshold,
        )
        if result:
            best_match_name, score, _ = result

            # --- BUG 修复点 ---
            # 1. 构造匹配文件的完整路径
            best_match_path = search_dir / best_match_name
            # 2. 从完整路径计算出相对于 base_dir 的正确相对路径
            #    这是修复目录重复问题的关键
            relative_path_for_include = best_match_path.relative_to(
                self.base_dir, walk_up=True
            )

            if score < 100:
                print(
                    f"INFO: 模糊查找 '{approximate_path}' -> 匹配到 '{relative_path_for_include}' (Partial Ratio 评分: {score:.2f}%)"
                )
            # 3. 使用修正后的相对路径调用 _include_file
            return self._include_file(
                str(relative_path_for_include), show_filename=True, tag=tag
            )
        else:
            raise FileNotFoundError(
                f"模糊查找失败: 无法为 '{approximate_path}' 在目录中找到超过阈值 ({similarity_threshold}%) 的匹配项。"
            )

    def _include_dir(
        self,
        path: str,
        glob_pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        tag: str = DEFAULT_DIR_TAG,
    ) -> str:
        """
        嵌入一个目录下的多个文件，并提供强大的过滤能力。
        """
        dir_path = self._resolve_path(path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"目录未找到: {dir_path}")
        exclude_patterns = exclude_patterns or []
        exclude_keywords = exclude_keywords or []
        all_files = list(dir_path.glob(glob_pattern))
        included_files_content = []
        for file_path in sorted(all_files):
            if not file_path.is_file():
                continue
            if any(file_path.match(p) for p in exclude_patterns):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                if any(keyword in content for keyword in exclude_keywords):
                    continue
            except (UnicodeDecodeError, IOError):
                print(
                    f"WARNING: Skipping non-text or unreadable file: {file_path}",
                    file=sys.stderr,
                )
                continue
            relative_path_for_display = file_path.relative_to(
                self.base_dir, walk_up=True
            )
            included_files_content.append(
                self._include_file(str(relative_path_for_display))
            )
        final_content = "\n".join(included_files_content)
        if tag.lower() == "none":
            return final_content
        path_attr = f' path="{path}"'
        return f"\n<{tag}{path_attr}>\n{final_content}\n</{tag}>\n"

    def render(self, template_path_str: str, **kwargs) -> str:
        """
        渲染指定的模板文件。
        """
        template_path = Path(template_path_str).resolve()
        if not template_path.is_file():
            raise FileNotFoundError(f"模板文件未找到: {template_path}")
        self._initialize_environment(template_path)
        if self.env is None:
            raise RuntimeError("Jinja2 environment not initialized.")
        template = self.env.from_string(remove_pattern(open(template_path).read()))
        prompt = template.render(**kwargs)
        return prompt

    def get_token_report(self, total_tokens: int, top_n: int = 5) -> str:
        """
        生成 token 使用情况的报告字符串。
        """
        if not self.token_stats:
            return "--- Token Usage Report ---\nNo files were included via include functions."
        report_lines = [f"--- Token Usage Report (Top {top_n}) ---"]
        sorted_stats = sorted(
            self.token_stats.items(), key=lambda item: item[1], reverse=True
        )
        max_path_len = (
            max(len(path) for path, _ in sorted_stats) if sorted_stats else 20
        )
        col_width = min(max(max_path_len, 20), 60)
        header = f"{'Tokens':>10} | {'Percentage':>12} | {'File Path':<{col_width}}"
        report_lines.append(header)
        for path, count in sorted_stats[:top_n]:
            percentage_str = (
                f"{(count / total_tokens) * 100:5.2f}%" if total_tokens > 0 else "N/A"
            )
            line = f"{count:>10,} | {percentage_str:>12} | {path:<{col_width}}"
            report_lines.append(line)
        if len(sorted_stats) > top_n:
            report_lines.append(f"... and {len(sorted_stats) - top_n} more files.")
        return "\n".join(report_lines)


def main():
    """
    主执行函数。
    """
    parser = argparse.ArgumentParser(
        description="一个智能的提示词（Prompt）生成引擎。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("template_file", type=str, help="Jinja2 模板文件的路径 (.j2)。")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.txt",
        help="保存结果的输出文件路径 (默认: output.txt)。",
    )
    parser.add_argument(
        "--no-clipboard", action="store_true", help="不将结果复制到系统剪贴板。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"用于 token 计算的模型 (默认: {DEFAULT_MODEL})。",
    )
    args = parser.parse_args()

    try:
        assembler = PromptAssembler(model=args.model)
        final_prompt = assembler.render(args.template_file)
        total_tokens = assembler._calc_token(final_prompt)
        token_report = assembler.get_token_report(total_tokens)
        print(token_report)
        print(f"✅ Total Tokens: {total_tokens:,}")
        if not args.no_clipboard:
            try:
                pyperclip.copy(final_prompt)
                print("✅ 结果已经复制到剪贴板")
            except pyperclip.PyperclipException:
                print(
                    "WARNING: 无法访问剪贴板。在无头服务器 (headless server) 上运行时这很常见。",
                    file=sys.stderr,
                )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_prompt)
        print(f"✅ 结果已经保存到 ./{args.output}")
    except Exception as e:
        print(f"\n发生未知错误: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
