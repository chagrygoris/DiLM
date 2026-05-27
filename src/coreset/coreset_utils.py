import torch
from datasets import Dataset
from torch.cuda import amp
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer


def batch_to_cuda(batch: dict[str, torch.Tensor] | BatchEncoding):
    """Load batch on cuda device"""
    return {k: v.cuda() for k, v in batch.items()}


def l2_dist(src: torch.Tensor, tgt: torch.Tensor):
    """Compute L2 distance
    Args:
        src (torch.Tensor): Source tensor of shape (n, d)
        tgt (torch.Tensor): Target tensor of shape (d,)
    Returns:
        dists (torch.Tensor): L2 distance of shape (n,)
    """
    return (src - tgt.unsqueeze(0)).pow(2).sum(1).pow(0.5)


def get_embeddings(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sentence_keys: list[str],
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute embeddings of training examples with encoder model"""

    model.cuda()
    model.eval()

    def _get_embedding(batch):
        sentences = tuple(batch[key] for key in sentence_keys)
        inputs = tokenizer(
            *sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.inference_mode():
            with amp.autocast(dtype=torch.bfloat16):
                # Only the final layer's CLS embedding is consumed, so do not ask
                # the model to retain all 13 hidden states (saves ~2 GB of VRAM
                # at batch_size=256 for bert-base, preventing OOM during the
                # post-training coreset step on smaller GPUs like Kaggle T4).
                outputs = model(**batch_to_cuda(inputs))
            embeddings = outputs.last_hidden_state[:, 0].cpu()
        return {"embedding": embeddings}

    embed_dataset = dataset.map(_get_embedding, batched=True, batch_size=batch_size)

    return torch.tensor(embed_dataset["embedding"])
