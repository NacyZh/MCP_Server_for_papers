import json

from rag.finetune.train_hyde_lora import _load_rows, _write_alpaca_dataset, _write_train_yaml


class Args:
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = ""
    lora_rank = 16
    lora_alpha = 32
    lora_target = "all"
    template = "qwen"
    cutoff_len = 1024
    max_samples = None
    batch_size = 1
    grad_accum = 8
    lr = 2e-4
    epochs = 2.0
    bf16 = False
    fp16 = False


def test_train_hyde_lora_writes_llamafactory_files(tmp_path):
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps({"query": "SCMA detection", "passage": "A paper-like passage."}) + "\n",
        encoding="utf-8",
    )
    rows = _load_rows(train_jsonl)
    dataset_dir = tmp_path / "data"
    dataset_path = _write_alpaca_dataset(rows, dataset_dir, "hyde_train")

    args = Args()
    args.output_dir = str(tmp_path / "out")
    yaml_path = _write_train_yaml(args, dataset_dir, "hyde_train")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    yaml_text = yaml_path.read_text(encoding="utf-8")

    assert dataset[0]["input"].startswith("Query: SCMA detection")
    assert dataset[0]["output"] == "A paper-like passage."
    assert info["hyde_train"]["formatting"] == "alpaca"
    assert "model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct" in yaml_text
    assert "finetuning_type: lora" in yaml_text
