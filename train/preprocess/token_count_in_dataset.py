from datasets import load_dataset
from transformers import AutoTokenizer


def count_tokens(dataset_path, tokenizer_name, text_column="text", split="train", num_rows=None):
    """
    统计 Hugging Face 数据集中所有文本的 token 数量，可限制检测的行数。

    Args:
        dataset_path (str): 数据集名称，例如 "Skywork/SkyPile-150B".
        tokenizer_name (str): 分词器名称，例如 "bert-base-chinese".
        text_column (str): 数据集中包含文本的列名，默认为 "text".
        split (str): 数据集分割名称，例如 "train".
        num_rows (int): 要检测的行数，默认为 None（检测全部数据）.

    Returns:
        int: 数据集中的总 token 数量.
    """
    # 加载数据集
    dataset = load_dataset('json', data_files=dataset_path, split=split)

    # 限制行数
    if num_rows is not None:
        dataset = dataset.select(range(min(num_rows, len(dataset))))

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 统计 token 数量
    def count_tokens_in_example(example):
        tokens = tokenizer(example[text_column], truncation=True, max_length=512)["input_ids"]
        return {"token_count": len(tokens)}

    # 应用映射函数
    tokenized_dataset = dataset.map(count_tokens_in_example, batched=False)

    # 计算总 token 数
    total_tokens = sum(tokenized_dataset["token_count"])

    print(f"Dataset: {dataset_path}, Split: {split}, Rows: {num_rows or 'All'}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Total tokens: {total_tokens}")
    return total_tokens


if __name__ == "__main__":
    dataset_path = "./train/data/pretrain/medical_encyclopedia.json"
    tokenizer_name = "bert-base-chinese"  # 分词器名称
    total_tokens = count_tokens(dataset_path, tokenizer_name, "text")
