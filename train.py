import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
import torch.distributed as dist
from torch import nn, optim
from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler

agent2map_modal = 2
agent2agent_modal = 3

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return dist.get_rank() == 0


def train_epoch(data_loader, predictor, planner, optimizer, use_planning, diversity):
    epoch_loss = []
    epoch_prediction_loss = []
    epoch_plan_score_loss = []
    epoch_prediction_score_loss = []
    epoch_MR_plan = []
    epoch_MR_prediction = []
    # epoch_cov_agent_map = []
    # epoch_cov_agent_agent = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        map_lanes = batch[2].to(args.device)
        map_crosswalks = batch[3].to(args.device)
        ref_line_info = batch[4].to(args.device)
        ground_truth = batch[5].to(args.device)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

        # predict
        optimizer.zero_grad()
        plans, predictions, plan_scores, prediction_scores, cost_function_weights, agent_map, agent_agent = predictor(ego, neighbors, map_lanes, map_crosswalks)
        # plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(modes*modes)], dim=1)
        plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(agent2agent_modal*agent2map_modal)], dim=1)
        prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction = MFMA_loss(plan_trajs, predictions, plan_scores, prediction_scores, ground_truth, weights) # multi-future multi-agent loss
        loss = 0.5 * prediction_loss + plan_score_loss + prediction_score_loss

        # diversity_agent_map = cosine_diversity_loss(agent_map)
        # diversity_agent_agent = cosine_diversity_loss(agent_agent)
        # if diversity:
        #     loss += diversity_agent_map + diversity_agent_agent

        
        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, plan_scores, prediction_scores)

            planner_inputs = {
                "control_variables": plan.view(-1, 100), # initial control sequence
                "predictions": prediction, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

            final_values, info = planner.layer.forward(planner_inputs)
            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3]) 
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, plan_scores, prediction_scores)


        try:
            # loss backward
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 5)
            optimizer.step()
            
        except RuntimeError as e:
            if "singular" in str(e):
                print(f"警告: 遇到奇异矩阵，跳过当前 batch。错误信息: {e}")
                # 清除可能残留的梯度，防止影响下一个 batch
                optimizer.zero_grad() 
            else:
                # 如果是其他错误，仍然抛出
                raise e

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights, prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())
        epoch_prediction_loss.append(prediction_loss.item())
        epoch_plan_score_loss.append(plan_score_loss.item())
        epoch_prediction_score_loss.append(prediction_score_loss.item())
        epoch_MR_plan.append(MR_plan.item())
        epoch_MR_prediction.append(MR_prediction.item())
        # epoch_cov_agent_map.append(diversity_agent_map.item())
        # epoch_cov_agent_agent.append(diversity_agent_agent.item())


        # show loss
        current += batch[0].shape[0]
        # 假设你希望显示 3 行
        lines_to_print = [
            f"Train Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.8f}  {(time.time()-start_time)/current:>.8f}s/sample  ",
            f"prediction_loss: {np.mean(epoch_prediction_loss):>.8f}  plan_score_loss: {np.mean(epoch_plan_score_loss):>.8f}  prediction_score_loss: {np.mean(epoch_prediction_score_loss):>.8f} ",
            f"MR_plan: {np.mean(epoch_MR_plan):>.8f}  MR_prediction: {np.mean(epoch_MR_prediction):>.8f}",
            # f"diversity_agent_map: {np.mean(epoch_cov_agent_map):>.8f}  diversity_agent_agent: {np.mean(epoch_cov_agent_agent):>.8f}"
        ]

        if use_planning:
            lines_to_print.append(f"plan_loss: {plan_loss:.8f}  plan_cost: {plan_cost:.8f}")
            sys.stdout.write(f"\033[1A")  # 上移行数
        

        for line in lines_to_print:
            sys.stdout.write("\r" + line + " " * 50 + "\n")  # 补几个空格清除残留

        sys.stdout.write(f"\033[3A")  # 上移行数
        sys.stdout.flush()


    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    prediction_loss, plan_score_loss, prediction_score_loss = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5]), np.mean(epoch_metrics[:, 6])
    MR_plan, MR_prediction = np.mean(epoch_metrics[:, 7]), np.mean(epoch_metrics[:, 8])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE, prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction]
    if is_main_process():
        logging.info(f'\ntrain-loss: {np.mean(epoch_loss):>.8f}'
                    f'\nprediction_loss: {prediction_loss:.8f}, plan_score_loss: {plan_score_loss:.8f}, prediction_score_loss: {prediction_score_loss:.8f}'
                    f'\nplannerADE: {plannerADE:.8f}, plannerFDE: {plannerFDE:.8f}, predictorADE: {predictorADE:.8f}, predictorFDE: {predictorFDE:.8f}, MR_plan: {MR_plan:.8f}, MR_prediction: {MR_prediction:.8f}')
        
    return np.mean(epoch_loss), epoch_metrics

def valid_epoch(data_loader, predictor, planner, use_planning, diversity):
    epoch_loss = []
    epoch_prediction_loss = []
    epoch_plan_score_loss = []
    epoch_prediction_score_loss = []
    epoch_MR_plan = []
    epoch_MR_prediction = []
    # epoch_cov_agent_map = []
    # epoch_cov_agent_agent = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        map_lanes = batch[2].to(args.device)
        map_crosswalks = batch[3].to(args.device)
        ref_line_info = batch[4].to(args.device)
        ground_truth = batch[5].to(args.device)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)



        # predict
        with torch.no_grad():        
            plans, predictions, plan_scores, prediction_scores, cost_function_weights, agent_map, agent_agent = predictor(ego, neighbors, map_lanes, map_crosswalks)
            # plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(modes*modes)], dim=1)
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(agent2map_modal*agent2agent_modal)], dim=1)
            prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction = MFMA_loss(plan_trajs, predictions, plan_scores, prediction_scores, ground_truth, weights) # multi-future multi-agent loss
            loss = 0.5 * prediction_loss + plan_score_loss + prediction_score_loss
    
            # diversity_agent_map = cosine_diversity_loss(agent_map)
            # diversity_agent_agent = cosine_diversity_loss(agent_agent)
            # if diversity:
            #     loss += diversity_agent_map + diversity_agent_agent

        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, plan_scores, prediction_scores)

            planner_inputs = {
                "control_variables": plan.view(-1, 100), # initial control sequence
                "predictions": prediction, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)
            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)
            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3]) 
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, plan_scores, prediction_scores)

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights, prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())
        epoch_prediction_loss.append(prediction_loss.item())
        epoch_plan_score_loss.append(plan_score_loss.item())
        epoch_prediction_score_loss.append(prediction_score_loss.item())
        epoch_MR_plan.append(MR_plan.item())
        epoch_MR_prediction.append(MR_prediction.item())
        # epoch_cov_agent_map.append(diversity_agent_map.item())
        # epoch_cov_agent_agent.append(diversity_agent_agent.item())


        # show loss
        current += batch[0].shape[0]
        #sys.stdout.write(f"\rTrain Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        # 假设你希望显示 3 行
        lines_to_print = [
            f"Train Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.8f}  {(time.time()-start_time)/current:>.8f}s/sample ",
            f"prediction_loss: {np.mean(epoch_prediction_loss):>.8f}  plan_score_loss: {np.mean(epoch_plan_score_loss):>.8f}  prediction_score_loss: {np.mean(epoch_prediction_score_loss):>.8f} ",
            f"MR_plan: {np.mean(epoch_MR_plan):>.8f}  MR_prediction: {np.mean(epoch_MR_prediction):>.8f}",
            # f"diversity_agent_map: {np.mean(epoch_cov_agent_map):>.8f}  diversity_agent_agent: {np.mean(epoch_cov_agent_agent):>.8f}"
        ]

        if use_planning:
            lines_to_print.append(f"plan_loss: {plan_loss:.8f}  plan_cost: {plan_cost:.8f}")
            sys.stdout.write(f"\033[1A")  # 上移行数
        # move cursor up n lines to overwrite previous output


        for line in lines_to_print:
            sys.stdout.write("\r" + line + " " * 50 + "\n")  # 补几个空格清除残留

        sys.stdout.write(f"\033[3A")  # 上移行数
        sys.stdout.flush()


    # show metrics

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    prediction_loss, plan_score_loss, prediction_score_loss = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5]), np.mean(epoch_metrics[:, 6])
    MR_plan, MR_prediction = np.mean(epoch_metrics[:, 7]), np.mean(epoch_metrics[:, 8])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE, prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction]
    if is_main_process():
        logging.info(f'\nval-loss: {np.mean(epoch_loss):>.8f}'
                    f'\nprediction_loss: {prediction_loss:.8f}, plan_score_loss: {plan_score_loss:.8f}, prediction_score_loss: {prediction_score_loss:.8f}'
                    f'\nval-plannerADE: {plannerADE:.8f}, val-plannerFDE: {plannerFDE:.8f}, val-predictorADE: {predictorADE:.8f}, val-predictorFDE: {predictorFDE:.8f}, MR_plan: {MR_plan:.8f}, MR_prediction: {MR_prediction:.8f}')


    return np.mean(epoch_loss), epoch_metrics

def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+f"{args.name}.log")

    if is_main_process():
        logging.info("------------- {} -------------".format(args.name))
        logging.info("Batch size: {}".format(args.batch_size))
        logging.info("Learning rate: {}".format(args.learning_rate))
        logging.info("Use integrated planning module: {}".format(args.use_planning))
        logging.info("Use diversity loss: {}".format(args.diversity))
        logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up predictor
    predictor = Predictor(50).to(args.device)

    predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, args.device)
    else:
        planner = None
    
    # set up optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)

    """warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_epochs = 50
    main_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])"""
    


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    start_epoch = 0

    # resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            if is_main_process():
                logging.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            
            # 加载模型状态
            predictor.module.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 确定起始epoch
            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            else:
                start_epoch = checkpoint['epoch']  
            
            if is_main_process():
                logging.info(f"Resuming training from epoch {start_epoch + 1}")
                logging.info(f"Previous train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
                logging.info(f"Previous val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            if is_main_process():
                logging.error(f"Checkpoint file {args.resume} not found!")
            return
    
    # set up data loaders
    train_set = DrivingData(args.train_set+'/*')
    valid_set = DrivingData(args.valid_set+'/*')
    train_sampler = DistributedSampler(train_set, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)
    if is_main_process():
        logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(start_epoch, train_epochs):
        if is_main_process():
            logging.info(f"Epoch {epoch+1}/{train_epochs}\n\n\n")
        train_sampler.set_epoch(epoch)

        
        # train 
        if planner:
            if epoch < args.pretrain_epochs:
                args.use_planning = False
            else:
                args.use_planning = True         

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, args.use_planning, args.diversity)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, args.use_planning, args.diversity)

        # save to training log
        if is_main_process():
            log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
                'train-plannerADE': train_metrics[0], 'train-plannerFDE': train_metrics[1], 
                'train-predictorADE': train_metrics[2], 'train-predictorFDE': train_metrics[3],
                'train-prediction_loss': train_metrics[4], 'train-plan_score_loss': train_metrics[5], 'train-prediction_score_loss': train_metrics[6], 
                'train-MR_plan': train_metrics[7], 'train-MR_prediction': train_metrics[8],
                'val-plannerADE': val_metrics[0], 'val-plannerFDE': val_metrics[1], 
                'val-predictorADE': val_metrics[2], 'val-predictorFDE': val_metrics[3],
                'val-prediction_loss': val_metrics[4], 'val-plan_score_loss': val_metrics[5], 'val-prediction_score_loss': val_metrics[6], 
                'val-MR_plan': val_metrics[7], 'val-MR_prediction': val_metrics[8]}

            if epoch == 0:
                with open(f'./training_log/{args.name}/{args.name}.csv', 'w') as csv_file: 
                    writer = csv.writer(csv_file) 
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with open(f'./training_log/{args.name}/{args.name}.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        # torch.save(predictor.state_dict(), f'training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth')

                ########################################################################################
        # 保存完整的checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': predictor.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'args': args  # 保存训练参数
        }

        model_path = f'training_log/{args.name}/{args.name}_{epoch+1}_{val_metrics[0]:.4f}.pth'
        if is_main_process():
            torch.save(checkpoint, model_path)


        ########################################################################################

        # # save model at the end of epoch
        # model_path = f'training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth'
        # torch.save(predictor.state_dict(), model_path)
        
        if is_main_process():
            logging.info(f"Model saved: {model_path}\n")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train datasets')
    parser.add_argument('--valid_set', type=str, help='path to validation datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument('--pretrain_epochs', type=int, help='epochs of pretraining predictor', default=5)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=2e-4)
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    # parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--diversity', action="store_true", help='if use diversity loss', default=False)
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from', default=None)
    parser.add_argument('--resume_epoch', type=int, help='epoch to resume from (if not specified, will be inferred from checkpoint name)', default=None)
    parser.add_argument('--a2m_modal', type=int, help='modal of agent2map (default: 2)', default=2)
    parser.add_argument('--a2a_modal', type=int, help='modal of agent2agent (default: 3)', default=3)
    args = parser.parse_args()

    local_rank = setup_ddp()
    args.device = f"cuda:{local_rank}"

    # Run
    model_training()

    if torch.distributed.is_initialized():
        dist.destroy_process_group()
