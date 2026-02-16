from __future__ import annotations

"""Checkpoint producer / manager for small-scale LM fine-tuning tasks.

This module provides a light-weight wrapper around Huggingface 🤗 Transformers
`Trainer` to finetune a causal language model on a *very small* corpus (a
single training example) and save the resulting weights to disk.


Typical usage
-------------

>>> from adra.utils_rl.checkpoint_manager import produce_checkpoint, CheckpointConfig
>>> texts = [
...     "def hello_world():\n    print('Hello, world!')",  # one or many training sequences
... ]
>>> model, ckpt_dir = produce_checkpoint(texts, CheckpointConfig(model_name_or_path="gpt2", num_train_epochs=1))
>>> print(f"Checkpoint saved to {ckpt_dir}")

"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
# -----------------------------------------------------------------------------
# Optional TRL / PEFT imports (only used when requested via config)
# -----------------------------------------------------------------------------
from typing import TYPE_CHECKING

try:
    from trl import SFTTrainer, SFTConfig  # type: ignore
    from peft import LoraConfig, TaskType  # type: ignore
except ImportError:  # pragma: no cover
    # Defer import error until we actually try to instantiate SFTTrainer
    SFTTrainer = None  # type: ignore
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore

# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Hyper-parameters and I/O settings for checkpoint creation."""

    # Model & tokenizer
    model_name_or_path: str = "gpt2"

    # Training data processing
    max_seq_length: int = 8192  # truncate / pad each sample to this length (in tokens)

    # Optimiser settings
    learning_rate: float = 5e-5
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 1

    # I/O & logging
    output_base_dir: str = "model_checkpoints"
    output_model_name: Optional[str] = None  # Override model name used in output directory (default: use model_name_or_path)
    save_total_limit: int = 1  # keep at most N checkpoints (older ones are deleted)
    logging_steps: int = 1
    seed: int = 42

    # Mixed precision – enabled automatically if CUDA is available but can be overridden
    fp16: Optional[bool] = None

    # Run in bfloat16 instead of fp16 (preferred on A100/H100 etc.)
    bf16: Optional[bool] = None

    # LR scheduler warm-up steps (passed straight to HF/TRL TrainingArguments)
    warmup_steps: int = 0
    warmup_ratio: float = 0.00
    
    # LR scheduler type - options: "constant", "linear", "cosine", "polynomial", etc.
    # "constant" is recommended for single-epoch memorization tasks
    lr_scheduler_type: str = "constant"

    # Gradient accumulation
    gradient_accumulation_steps: int = 1  # effective global batch = per_device × world_size × this

    # Memory-saving option
    gradient_checkpointing: bool = False   # if True, enable torch.utils.checkpoint to trade compute for memory

    # Automatically resume training from the last checkpoint in output_dir when available.
    # True  -> auto-detect latest checkpoint inside output_dir
    # str   -> explicit path to checkpoint directory
    # False -> always start fresh
    resume_from_checkpoint: bool | str = True

    # Optional custom name for the checkpoint directory
    checkpoint_name: Optional[str] = None

    # ------------------------------------------------------------------
    # TRL / LoRA / chat-template extensions
    # ------------------------------------------------------------------
    use_trl: bool = False                 # if True, use TRL SFTTrainer instead of vanilla Trainer
    use_lora: bool = False                # if True, wrap model with LoRA (requires use_trl)
    lora_r: int = 16                      # LoRA rank
    lora_alpha: int = 32                  # LoRA alpha
    lora_dropout: float = 0.05            # LoRA dropout

    apply_chat_template: bool = False     # whether to format each text via tokenizer.apply_chat_template
    add_generation_prompt: bool = False   # passed to apply_chat_template; ignored if above is False
    
    # TRL-specific parameters
    packing: bool = False                 # whether to use packing for efficient training
    remove_unused_columns: bool = False   # whether to remove unused columns from dataset

    def resolve_fp16(self) -> bool:
        if self.fp16 is not None:
            return self.fp16
        return torch.cuda.is_available()

    def resolve_bf16(self) -> bool:
        if self.bf16 is not None:
            return self.bf16
        # Prefer bf16 automatically if the GPU advertises adequate support
        # (A100/H100 – `torch.cuda.is_bf16_supported()` is only on newer PyTorch)
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False


# -----------------------------------------------------------------------------
# CheckpointManager – orchestrates training & saving
# -----------------------------------------------------------------------------


class CheckpointManager:
    """Utility class to finetune a model on small corpora and save checkpoints."""

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Timestamped directory for this run – keeps things tidy and unique.
        timestamp = datetime.now().strftime("%Y%m%d")
        # Use output_model_name if provided, otherwise extract from model_name_or_path
        if config.output_model_name:
            model_stub = config.output_model_name
        else:
            model_stub = Path(config.model_name_or_path).name.replace("/", "_")
        # Always include the timestamp; if a custom checkpoint_name is given, append it for clarity
        if config.checkpoint_name:
            self.output_dir = Path(config.output_base_dir) / f"{model_stub}_{timestamp}_{config.checkpoint_name}"
        else:
            self.output_dir = Path(config.output_base_dir) / f"{model_stub}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Logging setup – every CheckpointManager has its own file handler so
        # that logs are persisted next to the checkpoint artefacts.
        # ------------------------------------------------------------------
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Avoid adding multiple handlers if multiple managers are constructed
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(self.output_dir / "train.log") for h in self.logger.handlers):
            # Stream handler (console)
            if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler(self.output_dir / "train.log")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)

        # Log config parameters for reproducibility
        self.logger.info("Initialising CheckpointManager with config: %s", config)

        # Load tokenizer & model.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        # Update tokenizer's model_max_length to match our config
        self.tokenizer.model_max_length = config.max_seq_length
        # Ensure we have a pad token (especially important for GPT-like models).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Instantiate model - keep it on CPU; Trainer/Accelerate will move/shard as needed.
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        # self.model.to(self.device)

        # Optionally enable gradient checkpointing to save GPU memory
        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # HF wrappers
    # ------------------------------------------------------------------

    def finetune(self, train_texts: Sequence[str] | Dataset) -> str:
        """Finetune the model on *train_texts* and save a checkpoint.

        Args:
            train_texts: A sequence of arbitrary strings OR a Huggingface Dataset to train on.
        Returns:
            Path (as str) to the directory containing the saved checkpoint.
        """
        # --------------------------------------------------------------
        # Decide backend: TRL SFTTrainer or vanilla HF Trainer
        # --------------------------------------------------------------

        if self.config.use_trl:
            if SFTTrainer is None:
                raise ImportError("TRL is not installed – install `trl` and `peft` to use use_trl=True.")

            # ------------------------------------------------------------------
            # Prepare dataset
            # ------------------------------------------------------------------
            if isinstance(train_texts, Dataset):
                self.logger.info("Fine-tuning with TRL on supplied Dataset (%d rows)", len(train_texts))
                train_dataset = train_texts
            else:
                self.logger.info("Fine-tuning with TRL on %d raw text samples", len(train_texts))
                processed = [self._maybe_apply_chat_template(t) for t in train_texts]
                train_dataset = Dataset.from_dict({"text": processed})

            # ------------------------------------------------------------------
            # PEFT / LoRA config (optional)
            # ------------------------------------------------------------------
            peft_cfg = None
            if self.config.use_lora:
                if LoraConfig is None:
                    raise ImportError("peft is not installed – install it or set use_lora=False.")
                peft_cfg = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )

            # ------------------------------------------------------------------
            # SFTConfig – TRL-specific TrainingArguments subclass
            # ------------------------------------------------------------------
            bf16_flag = self.config.resolve_bf16()
            fp16_flag = self.config.resolve_fp16()
            if bf16_flag and fp16_flag:
                # Prefer bf16 and disable fp16 to satisfy transformers validation
                fp16_flag = False

            training_args = SFTConfig(
                output_dir=str(self.output_dir),
                overwrite_output_dir=True,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                save_strategy="epoch",
                save_total_limit=self.config.save_total_limit,
                logging_steps=self.config.logging_steps,
                seed=self.config.seed,
                fp16=fp16_flag,
                bf16=bf16_flag,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type=self.config.lr_scheduler_type,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                gradient_checkpointing=self.config.gradient_checkpointing,
                # Additional TRL parameters for better truncation handling
                packing=self.config.packing,  # Use config setting for packing
                remove_unused_columns=self.config.remove_unused_columns,  # Use config setting
            )

            # ------------------------------------------------------------------
            # Instantiate SFTTrainer
            # ------------------------------------------------------------------
            # Monkeypatch AutoTokenizer to ensure max_seq_length is set correctly
            # This is necessary because SFTTrainer loads its own tokenizer during __init__
            import functools
            from transformers import AutoTokenizer as _OrigAutoTokenizer
            
            original_from_pretrained = _OrigAutoTokenizer.from_pretrained
            max_seq_len_to_use = self.config.max_seq_length
            
            @functools.wraps(original_from_pretrained)
            def patched_from_pretrained(*args, **kwargs):
                tokenizer = original_from_pretrained(*args, **kwargs)
                # Update model_max_length for tokenizers loaded during SFTTrainer init
                if hasattr(tokenizer, 'model_max_length'):
                    tokenizer.model_max_length = max_seq_len_to_use
                return tokenizer
            
            # Temporarily patch AutoTokenizer
            AutoTokenizer.from_pretrained = patched_from_pretrained
            
            try:
                trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    peft_config=peft_cfg,
                )
            finally:
                # Restore original method
                AutoTokenizer.from_pretrained = original_from_pretrained
            
            self.logger.info("SFTTrainer initialized with max_seq_length=%d", self.config.max_seq_length)
        else:
            # ------------------------------------------------------------------
            # Original Trainer path (unchanged)
            # ------------------------------------------------------------------

            if isinstance(train_texts, Dataset):
                self.logger.info("Starting fine-tuning on a Huggingface Dataset with %d rows", len(train_texts))
                train_dataset = train_texts
            else:
                self.logger.info("Starting fine-tuning on %d text samples", len(train_texts))
                train_dataset = self._prepare_dataset(list(train_texts))
                self.logger.debug("Tokenised dataset with %d rows", len(train_dataset))

            bf16_flag = self.config.resolve_bf16()
            fp16_flag = self.config.resolve_fp16()
            if bf16_flag and fp16_flag:
                fp16_flag = False

            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                overwrite_output_dir=True,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                save_strategy="epoch",
                save_total_limit=self.config.save_total_limit,
                logging_steps=self.config.logging_steps,
                seed=self.config.seed,
                fp16=fp16_flag,
                bf16=bf16_flag,
                warmup_ratio=self.config.warmup_ratio,
                gradient_checkpointing=self.config.gradient_checkpointing,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        # --------------------------------------------------------------
        # Resume logic – look for an existing checkpoint if requested
        # --------------------------------------------------------------

        resume_path = None
        if self.config.resume_from_checkpoint is not False:  # True or explicit path
            if isinstance(self.config.resume_from_checkpoint, str):
                resume_path = Path(self.config.resume_from_checkpoint)
            else:
                # auto-detect latest checkpoint in output_dir
                checkpoints = list(self.output_dir.glob("checkpoint-*"))
                if checkpoints:
                    # sort by modification time
                    checkpoints.sort(key=lambda p: p.stat().st_mtime)
                    resume_path = checkpoints[-1]

            if resume_path and resume_path.exists():
                self.logger.info("Resuming training from checkpoint %s", resume_path)

        # --------------------------------------------------------------
        # Common training + saving logic
        # --------------------------------------------------------------

        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)

        # ----------------------------------------------------------
        # Save checkpoint – handle FSDP models by gathering full SD
        # ----------------------------------------------------------
        self.logger.info("Training complete. Saving model to %s", self.output_dir)

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig, state_dict_type  # type: ignore
        except ImportError:
            FSDP = None  # type: ignore

        if FSDP is not None and isinstance(self.model, FSDP):
            # Gather full state dict on rank-0 then save so that AutoModel can reload later.
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_cfg):
                trainer.save_model(self.output_dir)
        else:
            trainer.save_model(self.output_dir)

        # Free GPU memory
        self.model.to("cpu")
        torch.cuda.empty_cache()

        return str(self.output_dir)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load a model from *checkpoint_path* (defaults to the last produced path)."""
        # In distributed runs only rank-0 has the full model file after FSDP gather.
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return None  # other ranks return None to avoid file-missing error

        if checkpoint_path is None:
            checkpoint_path = self.output_dir

        return AutoModelForCausalLM.from_pretrained(checkpoint_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_dataset(self, texts: List[str]) -> Dataset:
        """Tokenise *texts* into a 🤗 `datasets.Dataset` ready for the Trainer."""

        dataset = Dataset.from_dict({"text": texts})

        def _tokenise(batch):
            out = self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
            )
            # For causal LM the loss is computed on *labels* == *input_ids*.
            out["labels"] = out["input_ids"].copy()
            return out

        dataset = dataset.map(_tokenise, batched=True, remove_columns=["text"])
        return dataset

    # ------------------------------------------------------------------
    # Internal helpers – TRL / chat templating
    # ------------------------------------------------------------------

    def _maybe_apply_chat_template(self, text: str) -> str:
        """Format *text* using the model's chat template if requested in config."""
        if not self.config.apply_chat_template:
            return text

        # Minimal wrapper: treat the raw text as a single user message.
        messages = [{"role": "user", "content": text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=self.config.add_generation_prompt,
            )
        except AttributeError as err:
            raise RuntimeError(
                "Tokenizer does not support chat templating; set apply_chat_template=False or use a compatible tokenizer."
            ) from err


# -----------------------------------------------------------------------------
# Convenience function – functional interface around CheckpointManager
# -----------------------------------------------------------------------------

def produce_checkpoint(
    train_texts: Sequence[str],
    config: Optional[CheckpointConfig] = None,
) -> Tuple[AutoModelForCausalLM, str]:
    """Finetune a model on *train_texts* and return *(model, checkpoint_dir)*."""
    if config is None:
        config = CheckpointConfig()
    manager = CheckpointManager(config)
    ckpt_dir = manager.finetune(train_texts)
    # Only rank-0 needs to load the model (for further use); others return None.
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        model = None
    else:
        model = manager.load_model(ckpt_dir)
    return model, ckpt_dir


# -----------------------------------------------------------------------------
# Re-export for convenient imports e.g. `from utils_rl.checkpoint_manager import ...`
# -----------------------------------------------------------------------------

__all__ = [
    "CheckpointConfig",
    "CheckpointManager",
    "produce_checkpoint",
] 