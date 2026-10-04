import logging
import os
import torch
from gmt_runtime_distributed import install_launch_backend
install_launch_backend()
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import sys
from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config

from gtr.config import add_gtr_config
from gtr.data.custom_build_augmentation import build_custom_augmentation
from gtr.data.gtr_dataset_dataloader import build_gtr_train_loader
from gtr.data.gtr_dataset_dataloader import build_gtr_test_loader
from gtr.data.gtr_dataset_mapper import GMTDatasetMapper
from gtr.costom_solver import build_custom_optimizer
from gtr.evaluation.mot_evaluation import MOTEvaluator
from gtr.modeling.freeze_layers import check_if_freeze_model

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7,8,9")

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    for dataset_name in cfg.DATASETS.TEST:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        assert evaluator_type == "mot", evaluator_type
        evaluator = MOTEvaluator(dataset_name, cfg, False, output_folder)

        if not comm.is_main_process():
            continue

        torch.multiprocessing.set_sharing_strategy("file_system")
        mapper = GMTDatasetMapper(
            cfg, False, augmentations=build_custom_augmentation(cfg, False)
        )
        data_loader = build_gtr_test_loader(cfg, dataset_name, mapper)
        results = inference_on_dataset(model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results)
    return results

def do_train(cfg, model, resume=False):
    model = check_if_freeze_model(model, cfg)
    model.train()
    assert cfg.SOLVER.USE_CUSTOM_SOLVER
    optimizer = build_custom_optimizer(cfg, model)
    if comm.get_world_size() > 1 and os.environ.get("GMT_POST_BACKWARD_CPU") == "1":
        from gmt_runtime_post_backward import install_optimizer_hook
        install_optimizer_hook(optimizer, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    checkpoint_state = checkpointer.resume_or_load(
        cfg.MODEL.WEIGHTS, resume=resume,
    )
    # The loop below increments its local index before the optimizer step.
    # Checkpoints are written after that step, so resuming must begin at the
    # saved index (not saved_index + 1) to avoid skipping one update.
    start_iter = checkpoint_state.get("iteration", -1)
    if start_iter < 0:
        start_iter = 0
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    assert cfg.VIDEO_INPUT
    mapper = GMTDatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    data_loader = build_gtr_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        l_a_aver = 0
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items() if 'loss' in k)
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for k, loss in loss_dict_reduced.items() \
                if 'loss' in k)
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=True)
            storage.put_scalar(
                "loss_asso_smooth", l_a_aver)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()
            
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()
            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            if iteration>0 and iteration%500==0:
                # Store the loop position explicitly.  The custom training loop
                # does not use PeriodicCheckpointer, so without this field
                # --resume restores weights/optimizer but restarts the local
                # iteration counter at zero.
                checkpointer.save("model_{}".format(iteration), iteration=iteration)
        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_gtr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="centernet")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    if os.environ.get("GMT_TRAIN_PROGRESS") == "1" and not args.eval_only:
        from gmt_runtime_progress import install as install_progress
        install_progress(model, cfg.OUTPUT_DIR, comm.get_rank())
    if os.environ.get("GMT_CHECKPOINT_BACKBONE") == "1" and not args.eval_only:
        from gmt_runtime_checkpoint import install
        logger.info("Runtime activation checkpointing with deterministic cuDNN and default TF32: %s", install(model))
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    post_backward_cpu = distributed and os.environ.get("GMT_POST_BACKWARD_CPU") == "1"
    if post_backward_cpu:
        if os.environ.get("GMT_CPU_COLLECTIVES") != "1" or os.environ.get("GMT_SYNC_DDP_BUCKETS") == "1":
            raise RuntimeError("Post-backward averaging requires CPU collectives and no DDP hook")
        from gmt_runtime_post_backward import synchronize_model
        synchronize_model(model)
        logger.info("Using CPU gradient averaging after independent backward, before optimizer clipping")
    elif distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )


    if distributed and not post_backward_cpu and os.environ.get("GMT_CPU_COLLECTIVES") == "1":
        from gmt_runtime_distributed import install_cpu_bucket_hook
        install_cpu_bucket_hook(model)
        logger.info("Using CPU-staged Gloo reductions and averaged DDP gradients")
    if distributed and os.environ.get("GMT_SYNC_DDP_BUCKETS") == "1":
        from gmt_runtime_distributed import install_synchronous_bucket_hook
        install_synchronous_bucket_hook(model)
        logger.info("Using standard averaged DDP buckets with explicit completion fences")
    do_train(cfg, model, resume=args.resume)
    return None


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    args.eval_only = False #false->train true->test 

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
