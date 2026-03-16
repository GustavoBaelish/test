import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states


# def MFMA_loss(plans, predictions, scores,score_prediction, ground_truth, weights):
#     global best_plan_mode
#     global best_prediction_mode
    
#     #import ipdb; ipdb.set_trace()

#     predictions = predictions * weights.unsqueeze(1)
#     prediction_distance = torch.norm(predictions[:, :, :, 9::10, :2] - ground_truth[:, None, 1:, 9::10, :2], dim=-1)
#     plan_distance = torch.norm(plans[:, :, 9::10, :2] - ground_truth[:, None, 0, 9::10, :2], dim=-1)
#     prediction_distance = prediction_distance.mean(-1)
#     plan_distance = plan_distance.mean(-1)

#     best_plan_mode = torch.argmin(plan_distance, dim=-1) 
#     score_loss = F.cross_entropy(scores, best_plan_mode)
#     best_prediction_mode = torch.argmin(prediction_distance, dim= 1) 
#     #import ipdb; ipdb.set_trace()
#     score_prediction_loss = F.cross_entropy(score_prediction.view(-1,score_prediction.shape[2]), best_prediction_mode.view(-1))
    

#     best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
#     #best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_prediction_mode)])
#     # 创建索引张量
#     batch_idx = torch.arange(predictions.size(0), device=predictions.device)[:, None, None]
#     agent_idx = torch.arange(predictions.size(2), device=predictions.device)[None, :, None]
#     # 选择最佳模式
#     best_mode_prediction = predictions[batch_idx, best_prediction_mode.unsqueeze(-1), agent_idx]
#     best_mode_prediction = best_mode_prediction.squeeze(2)  # 形状: [4, 10, 50, 3]


#     prediction = torch.cat([best_mode_plan.unsqueeze(1), best_mode_prediction], dim=1)

#     prediction_loss: torch.tensor = 0
#     for i in range(prediction.shape[1]):
#         prediction_loss += F.smooth_l1_loss(prediction[:, i], ground_truth[:, i, :, :3])
#         prediction_loss += F.smooth_l1_loss(prediction[:, i, -1], ground_truth[:, i, -1, :3])
        
#     return prediction_loss ,score_loss ,score_prediction_loss

def MFMA_loss(plans, predictions, plan_scores, prediction_scores, ground_truth, weights):
    global best_plan_mode
    global best_prediction_mode

    predictions = predictions * weights.unsqueeze(1)
    prediction_distance = torch.norm(predictions[:, :, :, :, :2] - ground_truth[:, None, 1:, :, :2], dim=-1)
    plan_distance = torch.norm(plans[:, :, :, :2] - ground_truth[:, None, 0, :, :2], dim=-1)

    # Miss Rate (MR)
    # For each sample: check whether the minimum FDE among M modes is > 2.0 meters
    min_FDE_plan, _ = plan_distance[..., -1].min(dim=1)     # [B]
    miss_plan = (min_FDE_plan > 2.0).float()   # 1 = miss, 0 = hit
    MR_plan = miss_plan.mean()                 # scalar: miss rate for this batch

    batch, modes, agents = prediction_distance[..., -1].shape
    prediction_fde = prediction_distance[..., -1].permute(0, 2, 1)
    prediction_fde = prediction_fde.reshape(batch * agents, modes)
    # import ipdb; ipdb.set_trace()
    prediction_fde_mask = (prediction_fde != 0).any(dim=1)
    prediction_fde = prediction_fde[prediction_fde_mask]

    min_FDE_prediction, _ = prediction_fde.min(dim=1)     # [B]
    miss_prediction = (min_FDE_prediction > 2.0).float()   # 1 = miss, 0 = hit
    MR_prediction = miss_prediction.mean()                 # scalar: miss rate for this batch
    # import ipdb; ipdb.set_trace()



    prediction_distance = prediction_distance.mean(-1)
    plan_distance = plan_distance.mean(-1)
    best_plan_mode = torch.argmin(plan_distance, dim=-1) 
    plan_score_loss = F.cross_entropy(plan_scores, best_plan_mode)
    best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_plan_mode)])

    weights_modes = weights.any(dim=-1).any(dim=-1)
    #import ipdb; ipdb.set_trace()
    prediction_scores = prediction_scores*weights_modes.unsqueeze(1) #[B, M, A]
    best_prediction_mode = torch.argmin(prediction_distance, dim= 1)
    #import ipdb; ipdb.set_trace()
    target_scores = prediction_scores.permute(0, 2, 1).reshape(-1, prediction_scores.shape[1])
    target_labels = best_prediction_mode.view(-1)
    prediction_score_loss = F.cross_entropy(target_scores, target_labels)


    indices = best_prediction_mode[:, None, :, None, None].expand(-1, 1, -1, predictions.shape[3], predictions.shape[4])
    best_mode_predictions = torch.gather(predictions, dim=1, index=indices)
    best_mode_predictions = best_mode_predictions.squeeze(1)  # [32, 10, 50, 3]

    prediction = torch.cat([best_mode_plan.unsqueeze(1), best_mode_predictions], dim=1)
    prediction_loss: torch.tensor = 0
    for i in range(prediction.shape[1]):
        prediction_loss += F.smooth_l1_loss(prediction[:, i], ground_truth[:, i, :, :3])
        prediction_loss += F.smooth_l1_loss(prediction[:, i, -1], ground_truth[:, i, -1, :3])
        
    #import ipdb; ipdb.set_trace()
    return prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction

# def MFMA_loss(plans, predictions, scores, ground_truth, weights):
#     global best_mode

#     predictions = predictions * weights.unsqueeze(1)

#     #1.
#     plans_xy = plans[..., :2]  # [32, 64, 50, 2]
#     gt_xy_plan = ground_truth[:, None, 0, :, :2]  # [32, 1, 50, 2]
#     dist_plan = torch.norm(plans_xy - gt_xy_plan, dim=-1)  # [32, 64, 50]
#     ADE_plan = dist_plan.mean(dim=-1)  # [32, 64]
#     FDE_plan = dist_plan[..., -1]  # [32, 64]

#     pred_xy = predictions[..., :2]  # [32, 64, 10, 50, 2]
#     gt_xy_pred = ground_truth[:, None, 1:, :, :2]  # [32, 1, 10, 50, 2]
#     dist_pred = torch.norm(pred_xy - gt_xy_pred, dim=-1)  # [32, 64, 10, 50]
#     # ADE: 每个 agent 的平均误差
#     ADE_pred = dist_pred.mean(dim=-1)  # [32, 64, 10]
#     # FDE: 每个 agent 的末端误差
#     FDE_pred = dist_pred[..., -1]  # [32, 64, 10]
#     # 可以取所有 agent 的平均
#     ADE_mean_pred = ADE_pred.mean(dim=-1)  # [32, 64]
#     FDE_mean_pred = FDE_pred.mean(dim=-1)  # [32, 64]

#     ADE = ADE_plan + ADE_mean_pred
#     FDE = FDE_plan + FDE_mean_pred
#     score_matrix = ADE + FDE
#     neg_score_matrix = -score_matrix  # [B, M]
#     temperature = 0.3
#     prob_matrix = F.softmax(neg_score_matrix / temperature, dim=-1).detach()  # [B, M]

#     score_loss = F.kl_div(scores.log(), prob_matrix, reduction='batchmean')
#     best_mode = torch.argmax(prob_matrix, dim=-1)  # [32]

#     # print("score_matrix:", score_matrix)
#     # print("prob_matrix:", prob_matrix)
#     # print("scores:", scores)
#     # print("scores min:", scores.min().item())
#     # print("scores has nan:", torch.isnan(scores).any())
#     # print("prob_matrix has nan:", torch.isnan(prob_matrix).any())


#     #2.
#     # prediction_distance = torch.norm(predictions[:, :, :, 9::10, :2] - ground_truth[:, None, 1:, 9::10, :2], dim=-1)
#     # plan_distance = torch.norm(plans[:, :, 9::10, :2] - ground_truth[:, None, 0, 9::10, :2], dim=-1)
#     # prediction_distance = prediction_distance.mean(-1).sum(-1)
#     # plan_distance = plan_distance.mean(-1)

#     # best_mode = torch.argmin(plan_distance+prediction_distance, dim=-1) 
#     # score_loss = F.cross_entropy(scores, best_mode)


#     best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
#     best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])
#     prediction = torch.cat([best_mode_plan.unsqueeze(1), best_mode_prediction], dim=1)

#     prediction_loss: torch.tensor = 0
#     for i in range(prediction.shape[1]):
#         prediction_loss += F.smooth_l1_loss(prediction[:, i], ground_truth[:, i, :, :3])
#         prediction_loss += F.smooth_l1_loss(prediction[:, i, -1], ground_truth[:, i, -1, :3])
        
#     return prediction_loss, score_loss

def select_future(plans, predictions, plan_scores, prediction_scores):
    plan = torch.stack([plans[i, m] for i, m in enumerate(best_plan_mode)])
    indices = best_prediction_mode[:, None, :, None, None].expand(-1, 1, -1, predictions.shape[3], predictions.shape[4])
    prediction = torch.gather(predictions, dim=1, index=indices)
    prediction = prediction.squeeze(1)  # [32, 10, 50, 3]

    return plan, prediction

def motion_metrics(plan_trajectory, prediction_trajectories, ground_truth_trajectories, weights, prediction_loss, plan_score_loss, prediction_score_loss, MR_plan, MR_prediction):
    #import ipdb; ipdb.set_trace()
    prediction_trajectories = prediction_trajectories * weights
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - ground_truth_trajectories[:, 1:, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, weights[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, weights[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)


    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item(), prediction_loss.item(), plan_score_loss.item(), prediction_score_loss.item(), MR_plan.item(), MR_prediction.item()

def project_to_frenet_frame(traj, ref_line):
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x, y = traj[:, :, 0], traj[:, :, 1]
    s = 0.1 * (k[:, :, 0] - 200)
    l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r))
    sl = torch.stack([s, l], dim=-1)

    return sl

def project_to_cartesian_frame(traj, ref_line):
    k = (10 * traj[:, :, 0] + 200).long()
    k = torch.clip(k, 0, 1200-1)
    ref_points = torch.gather(ref_line, 1, k.view(-1, traj.shape[1], 1).expand(-1, -1, 3))
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x = x_r - traj[:, :, 1] * torch.sin(theta_r)
    y = y_r + traj[:, :, 1] * torch.cos(theta_r)
    xy = torch.stack([x, y], dim=-1)

    return xy

def bicycle_model(control, current_state):
    dt = 0.1 # discrete time period [s]
    max_delta = 0.6 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    L = 3.089 # vehicle's wheelbase [m]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_delta, max_delta) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * delta / L # use delta to approximate tan(delta)
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

def physical_model(control, current_state, dt=0.1):
    dt = 0.1 # discrete time period [s]
    max_d_theta = 0.5 # vehicle's change of angle limits [rad/s]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate
    y_0 = current_state[:, 1] # vehicle's y-coordinate
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    d_theta = control[:, :, 1].clamp(-max_d_theta, max_d_theta) # vehicle's heading change rate [rad/s]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)

    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

# def covariance_decorrelation_loss(agent_map):
#     # agent_map: [B, M, N, D]
#     B, M, N, D = agent_map.shape

#     # reshape成每个模态一个长向量
#     features = agent_map.reshape(B, M, N*D)  # [B, M, N*D]

#     # 去均值
#     features = features - features.mean(dim=2, keepdim=True)  # M维每行去均值

#     # 模态间协方差: [B, M, M]
#     cov = torch.bmm(features, features.transpose(1, 2)) / (N*D)

#     # 去掉对角元素
#     eye = torch.eye(M, device=cov.device)
#     loss = ((cov * (1 - eye)) ** 2).mean()

#     return loss

def cosine_diversity_loss(modes_feature):
    """
    modes_feature: [B, M, N, D]
        B: batch size
        M: 模态数
        N: agent 数
        D: 特征维度

    功能：
        计算每个 agent 在不同模态下的平均余弦相似度（越小越好），
        然后在 agent 维度上取平均。
    """
    modes_feature = modes_feature.permute(0, 2, 1, 3)
    B, N, M, D = modes_feature.shape

    modes_feature = F.normalize(modes_feature, dim=-1)  # [B, M, N, D]
    
    # 计算模态间相似度矩阵：直接用矩阵相乘进行向量化
    # 输出 [B, N, M, M]
    sim_matrix = torch.einsum(
        "bnid,bnjd->bnij", 
        modes_feature, modes_feature
    )

    # 构造 eye
    eye = torch.eye(M, device=modes_feature.device)[None, None, :, :]  # [1,1,M,M]

    # 去掉对角线
    loss = (sim_matrix.abs() * (1 - eye)).mean()

    return loss


def covariance_matrix_loss(modes_feature): #B, M, N, D
    loss = 0.
    B, M, N, D = modes_feature.shape
    for n in range(N):
        for i in range(M):
            zi = F.normalize(modes_feature[:, i, n, :], dim=-1)  # [B, D]
            for j in range(i+1, M):
                zj = F.normalize(modes_feature[:, j, n, :], dim=-1)
                c = zi.T @ zj / zi.size(0)      # [D, D]
                loss += (c ** 2).sum()
    return loss

