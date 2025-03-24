
import argparse
from datetime import datetime
from pyhocon import ConfigFactory
import yaml
import os
import pdb


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Exo2Ego-V."
    )   
    parser.add_argument("--config_file", type=str, help="Path to the configuration file")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default="",
        help="The validiation directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--nerf_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=6,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--demo_image",
        type=str,
        default="",
        help=("The input image of validation"),
    )
    parser.add_argument(
        "--ego_dir",
        type=str,
        default=None,
        help=("The input images of ego views"),
    )
    parser.add_argument(
        "--exo_dir",
        type=str,
        default=None,
        help=("The input images of exo views"),
    )

    parser.add_argument(
        "--exo_width",
        type=int,
        default=256,
        help=("The width of exo views"),
    )

    parser.add_argument(
        "--exo_height",
        type=int,
        default=256,
        help=("The height of exo views"),
    )
    parser.add_argument(
        "--exo_cropped_width",
        type=int,
        default=256,
        help=("The width of exo views"),
    )

    parser.add_argument(
        "--exo_cropped_height",
        type=int,
        default=256,
        help=("The height of exo views"),
    )

    parser.add_argument(
        "--ego_width",
        type=int,
        default=256,
        help=("The width of ego views"),
    )

    parser.add_argument(
        "--ego_height",
        type=int,
        default=256,
        help=("The height of ego views"),
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help=("The num of frames"),
    )
    parser.add_argument(
        "--cat",
        type=bool,
        default=True,
        help=("whether to cat the latent and the conditional latent"),
    )
    parser.add_argument(
        "--action_num",
        type=int,
        default=0,
        help=("whether to cat the latent and the conditional latent"),
    )
    parser.add_argument(
        "--add_attention_layer",
        type=bool,
        default=False,
        help=("whether to add the self attention to get the final feature"),
    )
    parser.add_argument(
        "--add_dif_learning_rate",
        type=bool,
        default=False,
        help=("whether to add the self attention to get the final feature"),
    )
    parser.add_argument(
        "--learning_rate2",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--condition_ego",
        type=bool,
        default=False,
        help=("cross attention ego views"),
    )
    parser.add_argument(
        "--cross_multi",
        type=bool,
        default=False,
        help=("cross attention different frames"),
    )
    parser.add_argument(
        "--cat_ego",
        type=bool,
        default=False,
        help=("cat ego view"),
    )
    parser.add_argument(
        "--add_ego_noise",
        type=bool,
        default=False,
        help=("add ego noise"),
    )
    parser.add_argument(
        "--cat_ego_single",
        type=bool,
        default=False,
        help=("add ego noise"),
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help=("The output directory where the model predictions and checkpoints will be written."),
    )
    parser.add_argument('--data_list', nargs='+', help='str list')
    parser.add_argument('--validation_data_list', nargs='+', help='str list')
    parser.add_argument('--scene_list', nargs='+', help='str list')

    parser.add_argument(
        "--use_pixelnerf",
        type=bool,
        default=False,
        help=("add ego noise"),
    )

    parser.add_argument(
        "--only_validation",
        type=bool,
        default=False,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--simple_mlp",
        type=bool,
        default=False,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--render_whole",
        type=bool,
        default=False,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="",
        help=("The output directory where the model predictions and checkpoints will be written."),
    )
    parser.add_argument("--conf", "-c", type=str, default=None)
    parser.add_argument(
        "--only_ego",
        type=bool,
        default=False,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--use_fishereye",
        type=bool,
        default=False,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--logs_path", type=str, default="logs", help="logs output directory",
    )

    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="checkpoints output directory",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="visuals",
        help="visualization output directory",
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="visuals",
        help="visualization output directory",
    )
    parser.add_argument(
        "--nerf_path",
        type=str,
        default=None,
        help="visualization output directory",
    )
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument(
        "--exp_group_name",
        "-G",
        type=str,
        default=None,
        help="if we want to group some experiments together",
    )
    parser.add_argument(
        "--name", "-n", type=str, default="test", help="experiment name"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="4",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    parser.add_argument(
        "--use_KB",
        type=bool,
        default=False,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--ray_batch_size", "-R", type=int, default=128, help="Ray batch size"
    )
    parser.add_argument(
        "--add_self_attention",
        type=bool,
        default=False,
        help=("whether to add the self attention to get the final feature"),
    )
    parser.add_argument(
        "--exo_ego_train",
        type=bool,
        default=False,
        help=("whether to add the self attention to get the final feature"),
    )
    parser.add_argument(
        "--dataset_format",
        "-F",
        type=str,
        default=None,
        help="Dataset format, multi_obj | dvr | dvr_gen | dvr_dtu | srn",
    )
    parser.add_argument(
        "--datadir", "-D", type=str, default=None, help="Dataset directory"
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.1,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=6,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--freeze_enc",
        type=bool,
        default=False,
        help=("whether to add the self attention to get the final feature"),
    )
    parser.add_argument(
        "--only_nerf_training",
        type=int,
        default=2,
        help=(""),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help=("The output directory where the model predictions and checkpoints will be written."),
    )
    parser.add_argument("--config", type=str, default="./configs/training/stage1.yaml")

    
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in config.items():
        setattr(args, key, value)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    if args.exp_group_name is not None:
        args.logs_path = os.path.join(args.logs_path, args.exp_group_name)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.exp_group_name)
        args.visual_path = os.path.join(args.visual_path, args.exp_group_name)

    conf = ConfigFactory.parse_file(args.conf)
    if args.simple_mlp:
         conf["model"]['type'] = 'pixelnerf2'
    if args.dataset_format is None:
        args.dataset_format = conf.get_string("data.format", "dvr")

    return args,conf