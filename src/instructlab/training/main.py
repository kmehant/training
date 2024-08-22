# Standard
import argparse
import os
import subprocess
import sys
import json
from typing import Dict
import tempfile

# Third Party
from tuning.sft_trainer import parse_arguments, get_parser, train
from tuning.config.tracker_configs import (
    TrackerConfigFactory,
)
from torch.cuda import OutOfMemoryError
from huggingface_hub.utils._validators import HFValidationError


# First Party
from instructlab.training import config
from instructlab.training.async_logger import AsyncStructuredLogger
from instructlab.training.config import (
    DataProcessArgs,
    TorchrunArgs,
    TrainingArgs,
)
from instructlab.training.tokenizer_utils import setup_tokenizer
from instructlab.training.utils import (
    StreamablePopen,
    retrieve_chat_template,
    set_random_seed,
    setup_logger,
)
import instructlab.training.data_process as dp


def get_fsdp_config():
    return {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
        "fsdp_cpu_ram_efficient_loading": "True",
        "fsdp_forward_prefetch": "False",
        "fsdp_offload_params": "False",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_sync_module_states": "True",
        "fsdp_use_orig_params": "False",
    }


def prepare_config(args):
    # TODO
    # last_step
    # save_last​ : no turning off
    # log_level
    # is_granite​

    # add some additional flags for log parity
    # add quantization from fms-acceleration

    config = {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler,
        "warmup_steps": args.num_warmup_steps,
        "seed": args.seed,
        "fsdp": f"{args.sharding_strategy.lower()} auto_wrap",
        "r": args.lora_r,
        "peft_method": "lora" if args.lora_r > 0 else "none",
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.lora_target_modules,
        "use_flash_attn": not args.disable_flash_attn,
        "training_data_path": args.data_path,
        "fsdp_config": get_fsdp_config(),
        "torch_dtype": "bfloat16",
        "multipack": [args.effective_batch_size, args.max_batch_len],
        "padding_free": ["huggingface-injected", "residual", 0.1],
        "mlp_dropout": ["residual", 0.1],
        "loss_across_gpus": ["mean", "token"],
        "emb_dropout": ["inputs", 0.1],
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_steps": args.save_samples // args.effective_batch_size,
        "gradient_checkpointing": True,  # default enabled in main_ds.py
        "bf16": True,
        "adam_beta2": 0.95,
        "max_grad_norm": 1,
        "log_level": args.log_level.lower(),
    }
    if args.lora_quant_bits:
        config["bnb_qlora"] = ["nf4"]
    
    if args.cpu_offload_optimizer:
        config["fsdp"] = "full_shard offload auto_wrap"
        config["fsdp_config"]["fsdp_offload_params"] = "True"

    return config


def _train(args, tokenizer, metric_logger):
    parser = get_parser()
    job_config = prepare_config(args)
    print(job_config)
    try:
        (
            model_args,
            data_args,
            training_args,
            trainer_controller_args,
            tune_config,
            file_logger_config,
            aim_config,
            quantized_lora_config,
            fusedops_kernels_config,
            padding_free_config,
            exp_metadata,
        ) = parse_arguments(parser, job_config)

    except Exception as e:  # pylint: disable=broad-except
        print(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(1)

    # extra metadata passed via client
    metadata = None
    if exp_metadata is not None:
        try:
            metadata = json.loads(exp_metadata)
            if metadata is None or not isinstance(metadata, Dict):
                print("metadata cannot be converted to simple k:v dict ignoring")
                metadata = None
        except ValueError as e:
            print("failed while parsing extra metadata. pass a valid json %s", repr(e))

    combined_tracker_configs = TrackerConfigFactory()

    combined_tracker_configs.file_logger_config = file_logger_config
    combined_tracker_configs.aim_config = aim_config
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer.save_pretrained(tempdir)
        model_args.tokenizer_name_or_path = tempdir
        try:
            train(
                model_args=model_args,
                data_args=data_args,
                train_args=training_args,
                peft_config=tune_config,
                trainer_controller_args=trainer_controller_args,
                tracker_configs=combined_tracker_configs,
                additional_callbacks=None,
                exp_metadata=metadata,
                quantized_lora_config=quantized_lora_config,
                fusedops_kernels_config=fusedops_kernels_config,
                fast_attention_config=padding_free_config,
            )
        except (MemoryError, OutOfMemoryError) as e:
            print(f"OOM error during training. {e}")
            sys.exit(1)
        except FileNotFoundError as e:
            print("Unable to load file: {}".format(e))
            sys.exit(1)
        except HFValidationError as e:
            print(f"There may be a problem with loading the model. Exception: {e}")
            sys.exit(1)
        except (TypeError, ValueError, EnvironmentError) as e:
            print(
                f"Exception raised during training. This may be a problem with your input: {e}"
            )
            sys.exit(1)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Unhandled exception during training: {e}")
            sys.exit(1)


def main(args):
    # Third Party
    import yaml

    metric_logger = AsyncStructuredLogger(
        args.output_dir
        + f"/training_params_and_metrics_global{os.environ['RANK']}.jsonl"
    )
    if os.environ["LOCAL_RANK"] == "0":
        print(f"\033[38;5;120m{yaml.dump(vars(args), sort_keys=False)}\033[0m")
        metric_logger.log_sync({"script_params": vars(args)})

    setup_logger(args.log_level)
    CHAT_TEMPLATE, SPECIAL_TOKENS = retrieve_chat_template(args.chat_tmpl_path)
    tokenizer = setup_tokenizer(args.model_name_or_path, SPECIAL_TOKENS, CHAT_TEMPLATE)

    _train(args, tokenizer, metric_logger)


# public API
def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    """
    # early validation logic here
    if train_args.max_batch_len < train_args.max_seq_len:
        raise ValueError(
            f"the `max_batch_len` cannot be less than `max_seq_len`: {train_args.max_batch_len=} < {train_args.max_seq_len=}"
        )

    # process the training data
    if not os.path.exists(train_args.data_output_dir):
        os.makedirs(train_args.data_output_dir, exist_ok=True)
    dp.main(
        DataProcessArgs(
            # XXX(osilkin): make a decision here, either:
            #   1. the CLI is fully responsible for managing where the data is written
            #   2. we never cache it and simply write it to a tmp file every time.
            #
            # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
            # where the user has a defined place for new temporary data to be written.
            data_output_path=train_args.data_output_dir,
            model_path=train_args.model_path,
            data_path=train_args.data_path,
            max_seq_len=train_args.max_seq_len,
            chat_tmpl_path=train_args.chat_tmpl_path,
        )
    )

    if not os.path.exists(train_args.ckpt_output_dir):
        os.makedirs(train_args.ckpt_output_dir, exist_ok=True)
    command = [
        "torchrun",
        f"--nnodes={torch_args.nnodes}",
        f"--node_rank={torch_args.node_rank}",
        f"--nproc_per_node={torch_args.nproc_per_node}",
        f"--rdzv_id={torch_args.rdzv_id}",
        f"--rdzv_endpoint={torch_args.rdzv_endpoint}",
        __file__,
        f"--model_name_or_path={train_args.model_path}",
        f"--data_path={train_args.data_output_dir}/data.jsonl",
        f"--output_dir={train_args.ckpt_output_dir}",
        f"--num_epochs={train_args.num_epochs}",
        f"--effective_batch_size={train_args.effective_batch_size}",
        f"--learning_rate={train_args.learning_rate}",
        f"--num_warmup_steps={train_args.warmup_steps}",
        f"--save_samples={train_args.save_samples}",
        f"--log_level=INFO",
        f"--max_batch_len={train_args.max_batch_len}",
        f"--seed={train_args.random_seed}",
        f"--chat-tmpl-path={train_args.chat_tmpl_path}",
    ]

    if train_args.checkpoint_at_epoch:
        command.append("--checkpoint_at_epoch")

    if train_args.mock_data:
        command.append("--mock_data")
        if train_args.mock_len:
            command.append(f"--mock_len={train_args.mock_len}")

    if train_args.is_padding_free:
        command.append("--is_granite")

    if train_args.disable_flash_attn:
        if train_args.is_padding_free:
            raise RuntimeError(
                "ERROR: Trying to use padding-free transformer without flash attention is not supported"
            )
        command.append("--disable_flash_attn")

    if train_args.lora:
        command.extend(
            [
                f"--lora_r={train_args.lora.rank}",
                f"--lora_alpha={train_args.lora.alpha}",
                f"--lora_dropout={train_args.lora.dropout}",
                "--lora_target_modules",
            ]
        )
        command.extend(train_args.lora.target_modules)
        # hard-code 4-bit quantization for now, change this when we add more
        quant_dtype = train_args.lora.quantize_data_type
        quantization_is_enabled = quant_dtype in (
            config.QuantizeDataType.NF4,
            config.QuantizeDataType.NF4.value,
        )
        if quantization_is_enabled:
            command.append("--lora_quant_bits=4")

    # deepspeed opts
    if train_args.deepspeed_options.save_samples:
        command.append(f"--save_samples_ds={train_args.deepspeed_options.save_samples}")
    if train_args.deepspeed_options.cpu_offload_optimizer:
        command.extend(
            [
                "--cpu_offload_optimizer",
                f"--cpu_offload_optimizer_ratio={train_args.deepspeed_options.cpu_offload_optimizer_ratio}",
            ]
        )
        if train_args.deepspeed_options.cpu_offload_optimizer_pin_memory:
            command.append("--cpu_offload_optimizer_pin_memory")

    print(f"\033[92mRunning command: {' '.join(command)}\033[0m")
    process = None
    try:
        process = StreamablePopen(
            f"{train_args.ckpt_output_dir}/full_logs_global{torch_args.node_rank}.log",
            command,
        )

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if "process" not in locals() or process is None:
            return
        if process.poll() == 0:
            print("\033[92mOperation completed successfully! 🎉\033[0m")
        else:
            print("\033[91mOperation failed, terminating process.\033[0m")

        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("\033[91mProcess did not terminate in time, killing it.\033[0m")
            process.kill()


if __name__ == "__main__":
    # TODO(osilkin): Configure a type that these args must adhere to for the sake of type checking
    #               Maybe switch out from argparse to something smarter
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--last_step",
        type=int,
        default=0,
        help="understand this as the last completed step. "
        "The default is 0, since global_step starts from 1 by default.",
    )
    # parser.add_argument("--samples_per_gpu", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=3840)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--save_samples",
        type=int,
        help="The number of samples seen between each checkpoint save. If --save_samples<=0, this feature is disabled.",
    )
    parser.add_argument(
        "--save_samples_ds",
        type=int,
        help="for saving in ds native format",
        default=None,
    )
    parser.add_argument(
        "--save_last", action="store_true", help="save after finishing training"
    )
    parser.add_argument(
        "--checkpoint_at_epoch",
        action="store_true",
        help="Save a model checkpoint after finishing an epoch.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock_data", action="store_true")
    parser.add_argument("--mock_len", type=int, default=2600)
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        # choices=[e.name for e in ShardingStrategy],
        default="FULL_SHARD",
        help="Sharding strategy to be used for distributed training.",
    )
    parser.add_argument("--is_granite", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0)  # set to > 0 to activate lora
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_quant_bits", type=int, default=None)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    parser.add_argument("--max_batch_len", type=int, default=60000)
    parser.add_argument(
        "--cpu_offload_optimizer",
        action="store_true",
        default=False,
        help="Offload optimizer to CPU when using DeepSpeed. This configures it to use ZeRO stage 2.",
    )
    parser.add_argument(
        "--cpu_offload_optimizer_pin_memory",
        action="store_true",
        default=False,
        help="Pin memory when offloading optimizer to CPU. This allows for faster transfers between CPU and GPU. Comes at the cost of higher memory usage and CPU overhead.",
    )
    parser.add_argument(
        "--cpu_offload_optimizer_ratio",
        type=float,
        default=1.0,
        help="Ratio of the optimizer to be offloaded to CPU. The rest will be on GPU(s).",
    )
    parser.add_argument("--NEFTune_alpha", type=float, default=None)
    parser.add_argument(
        "--chat-tmpl-path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "chat_templates/ibm_generic_tmpl.py"
        ),
    )
    parser.add_argument("--disable_flash_attn", action="store_true")
    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)

"""
pkill python
git reset --hard
git pull
export WORLD_SIZE=1
sleep 3
mkdir -p /new_data/experiments/ap-fsdp-p00-old-m-ds-2t
cd /app/fsdp
export WORLD_SIZE=1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
--nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" main_ds.py \
--model_name_or_path=mistralai/Mistral-7B-v0.1 \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/new_data/experiments/ap-fsdp-p00-old-m-ds-2t" \
--num_epochs=100 \
--samples_per_gpu=24 \
--learning_rate=1e-06 \
--num_warmup_steps=800 \
--gradient_accumulation_steps=2 \
--save_samples=12000 \
--log_level="INFO" \
--mock_data \
--mock_len=2048 \
--seed=42 | tee /new_data/experiments/ap-fsdp-p00-old-m-ds-2t/$RANK.log
export WORLD_SIZE=1
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
--nproc_per_node=8 --rdzv_id=101 \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" main_ds.py \
--model_name_or_path=/new_data/models/granite7b/ibm_models_version/ \
--data_path="/dev/shm/data.jsonl" \
--output_dir="/new_data/experiments/ap-granite-4t" \
--num_epochs=100 \
--samples_per_gpu=240 \
--learning_rate=2e-05 \
--num_warmup_steps=385 \
--gradient_accumulation_steps=2 \
--save_samples=250000 \
--log_level="INFO" \
--sharding_strategy="HYBRID_SHARD" \
--is_granite \
--max_batch_len 70000 \
--seed=42
"""
