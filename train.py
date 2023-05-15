from absl import logging
import flax
import jax
from matplotlib import pyplot as plt
import numpy as np
import optax
import tqdm
import argparse

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import utils
from vit_jax import models
from vit_jax import train
from vit_jax.configs import common as common_config
from vit_jax.configs import models as models_config

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)
    parser.add_argument('--dataset', default='vit_acne04_ds', type=str, help='dataset name or directory')
    parser.add_argument('--batch-size', default=64, type=int, help='total batch size for all GPUs/CPUs')
    parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
    parser.add_argument('--pp_crop', default=100, type=int, help=' MAYBE train, val image size (pixels)')
    parser.add_argument('--warmup', default=5, type=int, help=' MAYBE train, val image size (pixels)')
    
    
    
    
    parser.add_argument('--conf-file', default='./configs/yolov6m_finetune.py', type=str, help='experiments description file')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=5, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
    parser.add_argument('--heavy-eval-range', default=50, type=int,
                        help='evaluating every epoch for last such epochs (can be jointly used with --eval-interval)')
    parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets')
    parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
    parser.add_argument('--output-dir', default='./runs/prohibited_object', type=str, help='path to save outputs')
    parser.add_argument('--project_name', default='prohibited_object', help='save to project/name')
    parser.add_argument('--name', default='YOLOv6m_with_COCO_DATASET_160223', help='save to project/name')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume the most recent training')
    parser.add_argument('--write_trainbatch_tb', action='store_true', help='write train_batch image to tensorboard once an epoch, may slightly slower train speed if open')
    parser.add_argument('--stop_aug_last_n_epoch', default=15, type=int, help='stop strong aug at last n epoch, neg value not stop, default 15')
    parser.add_argument('--save_ckpt_on_last_n_epoch', default=-1, type=int, help='save last n epoch even not best or last, neg value not save')
    parser.add_argument('--distill', action='store_true', help='distill or not')
    parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
    parser.add_argument('--quant', action='store_true', help='quant or not')
    parser.add_argument('--calib', action='store_true', help='run ptq')
    parser.add_argument('--teacher_model_path', type=str, default=None, help='teacher model path')
    parser.add_argument('--temperature', type=int, default=20, help='distill temperature')
    parser.add_argument('--fuse_ab', action='store_true', help='fuse ab branch in training process or not')
    parser.add_argument('--bs_per_gpu', default=32, type=int, help='batch size per GPU for auto-rescale learning rate, set to 16 for P6 models')
    return parser

def main(args):
    ##### LOAD DATASET #####
    dataset = args.dataset
    batch_size = args.batch_size
    num_classes = input_pipeline.get_directory_info(dataset)['num_classes']
    
    config = common_config.with_dataset(common_config.get_config(), dataset)
    config.batch = batch_size
    config.pp.crop = args.pp_crop
    input_pipeline.get_directory_info(dataset)
    
    # For details about setting up datasets, see input_pipeline.py on the right.
    # ds_train = input_pipeline.get_data_from_tfds(config=config, mode='train')
    ds_train = input_pipeline.get_data_from_directory(config=config, 
                                                    directory='/home/bach/Bach/Coding/Python/Hackathon2023/acne04/acne04_Classification/vit_ds_class',         
                                                    mode='train')
    # ds_test = input_pipeline.get_data_from_tfds(config=config, mode='test')
    ds_test = input_pipeline.get_data_from_directory(config=config, 
                                                    directory='/home/bach/Bach/Coding/Python/Hackathon2023/acne04/acne04_Classification/vit_ds_class',         
                                                    mode='test')
    
    ##### LOAD PRETRAINED MODEL #####
    model_name = 'ViT-B_32'
    model_config = models_config.MODEL_CONFIGS[model_name]
    
    # Load model definition & initialize random parameters.
    # This also compiles the model to XLA (takes some minutes the first time).
    batch = next(iter(ds_train.as_numpy_iterator()))
    if model_name.startswith('Mixer'):
        model = models.MlpMixer(num_classes=num_classes, **model_config)
    else:
        model = models.VisionTransformer(num_classes=num_classes, **model_config)
    variables = jax.jit(lambda: model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension of the batch for initialization.
        batch['image'][0, :1],
        train=False,
    ), backend='cpu')()
        
    # Load and convert pretrained checkpoint.
    # This involves loading the actual pre-trained model results, but then also also
    # modifying the parameters a bit, e.g. changing the final layers, and resizing
    # the positional embeddings.
    # For details, refer to the code and to the methods of the paper.
    params = checkpoint.load_pretrained(
        pretrained_path=f'{model_name}.npz',
        init_params=variables['params'],
        model_config=model_config,
    )
    
    ##### FINETUNE #####
    # 100 Steps take approximately 15 minutes in the TPU runtime.
    total_steps = args.epochs
    warmup_steps = args.warmup_steps
    decay_type = 'cosine'
    grad_norm_clip = 1
    # This controls in how many forward passes the batch is split. 8 works well with
    # a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
    # also adjust the batch_size above, but that would require you to adjust the
    # learning rate accordingly.
    accum_steps = 8
    base_lr = 0.03
    
    
    
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
