import torch
from torch import nn
import ipdb
import math
import torch.nn.functional as F
from utils.train_utils import bicycle_model

agent2map_modal = 2
agent2agent_modal = 3

class GatedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, query, key, value, key_padding_mask=None):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        B, L, D = query.size()
        S = key.size(1)

        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            # key_padding_mask: [B, S] -> Padding 的位置为 True
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # --- 核心修改：门控 (Sigmoid) 注意力 ---
        # 权重之和不为 1。元素可以被忽略 (0) 或被完全关注 (1)
        attn_weights = torch.sigmoid(scores)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        if not self.batch_first:
            out = out.transpose(0, 1)
            
        return out, attn_weights

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]

        return output
    
# Local context encoders
class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU())

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int()) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        stop_point = self.stop_point(inputs[..., 14].int())
        interpolating = self.interpolating(inputs[..., 15].int()) 
        stop_sign = self.stop_sign(inputs[..., 16].int())

        lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.pointnet(lane_embedding)

        return output

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
    
    def forward(self, inputs):
        output = self.point_net(inputs)

        return output

# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = GatedMultiheadAttention(256, 8, 0.1, batch_first=True)
        self.transformer = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 256), nn.LayerNorm(256))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.transformer(attention_output)

        return output #shape [B, 1, 256]


class MoEMultiModalTransformer(nn.Module):
    def __init__(self, num_experts, d_model=256, hidden_dim=1024):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        # 1. 门控网络 (Gating Network / Router)
        # 输入 Query 特征，输出每个专家的权重 (num_experts)
        self.gate = nn.Linear(d_model, num_experts)

        # 2. 专家网络 (Experts)
        # 保持你原来的结构：每个专家包含 Attn + FFN
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "attn": GatedMultiheadAttention(d_model, 4, dropout=0.1, batch_first=True),
                "ffn": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, d_model),
                    nn.LayerNorm(d_model)
                )
            }) for _ in range(num_experts)
        ])

        # 最后的 LayerNorm，保证融合后的分布稳定
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        """
        query: [B, L, D]
        key, value: [B, S, D]
        """
        # --- A. 计算门控权重 (Routing) ---
        # gate_logits: [B, L, num_experts]
        gate_logits = self.gate(query) 
        # 使用 Softmax 归一化，使得 expert_weights 之和为 1
        expert_weights = F.softmax(gate_logits, dim=-1) 

        # --- B. 计算所有专家的输出 ---
        expert_outputs = []
        for j in range(self.num_experts):
            # 1. Attention
            # attn_out: [B, L, D]
            attn_out, _ = self.experts[j]["attn"](query, key, value, key_padding_mask=mask)
            # 残差连接通常在这里做，或者在下面做，这里假设专家内部输出纯净的变换
            # 2. FFN
            ffn_out = self.experts[j]["ffn"](attn_out)
            expert_outputs.append(ffn_out)

        # 堆叠专家输出: [B, L, num_experts, D]
        # 注意维度变换，方便后续计算
        stacked_experts = torch.stack(expert_outputs, dim=2)

        # --- C. 加权融合 (Weighted Sum) ---
        # weights: [B, L, num_experts] -> 扩维变成 [B, L, num_experts, 1] 以便广播
        weights_expanded = expert_weights.unsqueeze(-1)
        
        # 融合: sum( weight_i * expert_output_i )
        # combined_output: [B, L, D]
        combined_output = torch.sum(weights_expanded * stacked_experts, dim=2)

        # --- D. 残差连接 + Norm (可选但推荐) ---
        # 将原始 query 加回去 (Residual Connection)
        output = self.output_norm(combined_output + query)

        return output


class MapFeature(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(MapFeature, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, 6, 100, 256]
        B, L, P, D = x.shape
        x = x.view(B, L * P, D)               # [B, 600, 256]
        attn = torch.sigmoid(self.attn(x))  # [B, 600, 1]
        map_feature = torch.sum(attn * x, dim=1)   # [B, 256]
        return map_feature



# Transformer-based encoders


class AgentFeature(nn.Module):
    def __init__(self):
        super(AgentFeature, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=4, enable_nested_tensor=False)

    def forward(self, inputs, mask=None):
        output = self.interaction_net(inputs, src_key_padding_mask=mask)

        return output

class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()
        self.crosswalk_attention = CrossTransformer()
        self.map_attention = MoEMultiModalTransformer(num_experts=agent2map_modal) 
        self.map_feature = MapFeature()

    def forward(self, actor, lanes, crosswalks, mask):
        #ipdb.set_trace()
        query = actor.unsqueeze(1)
        map_feature = self.map_feature(torch.cat([lanes, crosswalks], dim=1))
        lanes_actor = [self.lane_attention(query, lanes[:, i], lanes[:, i]) for i in range(lanes.shape[1])]
        crosswalks_actor = [self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i]) for i in range(crosswalks.shape[1])]
        map_actor = torch.cat(lanes_actor+crosswalks_actor, dim=1)
        #ipdb.set_trace()
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(1)
        #ipdb.set_trace()

        #return map_feature, output 
        return map_feature, output #shape [B, 256], [B, 1, 256]
        
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.multi_modal = MoEMultiModalTransformer(num_experts=agent2agent_modal)

    def forward(self, inputs, mask=None):
        feature = self.interaction_net(inputs, src_key_padding_mask=mask)
        output = self.multi_modal(feature, feature, feature, mask)
        #ipdb.set_trace()

        return output #shape [B, N, 256]


class AgentDecoder(nn.Module):
    def __init__(self, future_steps, num_modes=agent2map_modal*agent2agent_modal, hidden_dim=256):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        # Learnable Queries 
        # 这里的每一个向量初始化时是随机的，训练后会代表一种特定的“意图原型”
        self.mode_queries = nn.Parameter(torch.randn(num_modes, hidden_dim))
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = GatedMultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim) #用于残差连接
        
        # A. Trajectory Head: 解码轨迹增量 (dx, dy, dtheta)
        # 输入: [..., hidden_dim] -> 输出: [..., steps*3]
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, future_steps * 3)
        )

        self.traj_encoder = nn.Sequential(
            nn.Linear(future_steps * 3, 64), # 输入变成了 x, y, theta
            nn.ELU(),
            nn.Linear(64, 64)
        )


        # B. Score Head: 预测每个模态的概率
        # 输入: [B, num_modes, hidden_dim] -> 输出: [B, num_modes, 1]
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + hidden_dim*4, 256),
            nn.ELU(),
            nn.Linear(256, 1) # 输出 Logits
        )
        

    def transform(self, prediction, current_state):
        # prediction: [B, M, A, T, 3] (假设这是 permute 后的形状)
        # current_state: [B, A, 3] (x, y, theta)

        # 1. 扩展 current_state 维度以匹配 prediction
        # [B, A, 3] -> [B, 1, A, 1, 3] (增加 Mode=1, Time=1 维度)
        state_expanded = current_state.unsqueeze(1).unsqueeze(3) 

        x = state_expanded[..., 0]
        y = state_expanded[..., 1]
        theta = state_expanded[..., 2]

        # 2. 预测的是增量 (dx, dy, dtheta)
        delta_x = prediction[..., 0]
        delta_y = prediction[..., 1]
        delta_theta = prediction[..., 2]

        # 3. 广播相加
        new_x = x + delta_x
        new_y = y + delta_y
        new_theta = theta + delta_theta

        return torch.stack([new_x, new_y, new_theta], dim=-1)
       
    # def forward(self, agent_map, agent_agent, current_state, actors_history):
    def forward(self, agent_map, agent_agent, neighbor_actors, current_state, map_feature):
        # agent_map: [B, A, D] (单一模态)
        # agent_agent: [B, A, D] (单一模态)
        # map_feature: [B, A, D] (单一模态)

        B, A, D = agent_map.shape
        # Step 1: 构建 Agent 的上下文 (改用 Stack)
        # 1. Stack: [B, A, D] -> [B, A, 4, D] (在第2维堆叠)
        context_stack = torch.stack([agent_map, agent_agent, neighbor_actors, map_feature], dim=2)
        # 2. Reshape 合并 Batch 和 Agent: [B*A, 4, D]
        # 这样每一个 Agent 都被视为一个独立的样本，拥有一段长度为 4 的 Context 序列
        context_seq = context_stack.view(B * A, 4, D)
        context_seq = self.context_proj(context_seq) 

        # Step 2: 准备 Queries 并进行 Attention
        # Mode Queries: [num_modes, D]
        # 扩展 Queries 到每个样本: [B*A, num_modes, D]
        queries = self.mode_queries.unsqueeze(0).repeat(B * A, 1, 1)
        
        # Attention: 
        # Query: [B*A, Modes, D] (意图)
        # Key/Val: [B*A, 3, D] (环境序列)
        # 此时 Attention 会自动决定每个 Agent 的每个意图更关注 Map 还是 Social
        attn_output, _ = self.cross_attn(query=queries, key=context_seq, value=context_seq)
        mode_features = self.layer_norm(queries + attn_output) # [B*A, Modes, D]

        # Step 3: 解码轨迹 (Trajectory Generation)
        # [B*A, Modes, D] -> [B*A, Modes, T*3]
        flat_traj_deltas = self.traj_head(mode_features)
        # Reshape: [B, A, Modes, T, 3] -> permute -> [B, Modes, A, T, 3]
        traj_deltas_reshaped = flat_traj_deltas.view(B ,A, self.num_modes, self._future_steps, 3).permute(0, 2, 1, 3, 4)
        trajs = self.transform(traj_deltas_reshaped, current_state) # [B, M, A, T, 3]

        # Step 5: 打分 (Scoring)
        
        # A. 编码生成的轨迹
        # [B*A, Modes, T*3] -> [B*A, Modes, 64]
        traj_embed = self.traj_encoder(flat_traj_deltas)
        
        global_context = context_seq.view(B*A, 4*self.hidden_dim) # [B*A, 4*D]
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.num_modes, -1) #[B*A, 4*D] -> [B*A, Modes, 4*D]
        
        # C. 拼接: Intent + Traj + Map
        # [B*A, Modes, 256 + 64 + 256*4]
        score_input = torch.cat([
            mode_features,  # 意图 (Intent) [B*A, Modes, hidden_dim]
            traj_embed,     # 物理结果 (Physics Result) [B*A, Modes, 64]
            global_context_expanded  # 环境约束 (Map/Obstacles) [B*A, Modes, hidden_dim*4]
        ], dim=-1)
        
        # D. 计算 Logits
        scores_flat = self.score_head(score_input).squeeze(-1) # [B*A, Modes]
        
        # E. 还原维度
        # [B*A, Modes] -> [B, A, Modes] -> [B, Modes, A] (为了匹配你之前的输出格式)
        scores = scores_flat.view(B, A, self.num_modes).permute(0, 2, 1)
        # 这里返回的是 Logits，外面如果需要概率可以用 F.softmax(scores, dim=1)
        return trajs, scores
        
    
class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9, num_modes=agent2agent_modal*agent2map_modal, hidden_dim=256):
        """
        num_modes: 规划出的轨迹模态数量
        """
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self.num_modes = num_modes 
        self.hidden_dim = hidden_dim

        # Learnable Queries 
        # 这里的每一个向量初始化时是随机的，训练后会代表一种特定的“意图原型”
        self.mode_queries = nn.Parameter(torch.randn(num_modes, hidden_dim))
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = GatedMultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim) #用于残差连接
        
        # A. Control Head: 解码控制信号
        # 输入: [B, num_modes, hidden_dim] -> 输出: [B, num_modes, steps*2]
        # 使用 MLP 逐步解码，比单层 Linear 表达能力更强
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, future_steps * 2) 
        )

        # 新增 Trajectory Encoder (轨迹编码器)
        # 作用：把生成的 [50, 3] 的轨迹压缩成特征向量，让 Score Head 能“看懂”轨迹
        self.traj_encoder = nn.Sequential(
            nn.Linear(future_steps * 3, 64), # 输入变成了 x, y, theta
            nn.ELU(),
            nn.Linear(64, 64)
        )
        
        # B. Score Head: 预测每个模态的概率
        # 输入: [B, num_modes, hidden_dim] -> 输出: [B, num_modes, 1]
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + hidden_dim*4, 256),
            nn.ELU(),
            nn.Linear(256, 1) # 输出 Logits
        )
        # C. Cost Function Weights Head (保持基于全局特征)
        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128), # 假设输入是 combined + map_feature
            nn.ReLU(),
            nn.Linear(128, feature_len),
            nn.Softmax(dim=-1)
        )
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))


    def forward(self, agent_map, agent_agent, ego_actor, ego, map_feature):
        """
        agent_map: [B, D]
        agent_agent: [B, D]
        ego_actor: [B, D]
        map_feature: [B, D] (用于全局辅助)
        """
        B = agent_map.shape[0]

        # Step 1: 构建上下文 (Context Construction)
        # 我们不再把它们 concat 成一条长向量，而是 stack 成一个序列 (Sequence)
        # 这样 Attention 机制可以自动分配权重（比如某些模态更关注 Map，某些更关注 Agent）
        # Context Shape: [B, 4, hidden_dim]  (Seq_Len = 4: Map, Social, Ego, Global)
        context_seq = torch.stack([agent_map, agent_agent, ego_actor, map_feature], dim=1)
        context_seq = self.context_proj(context_seq) 
        queries = self.mode_queries.unsqueeze(0).repeat(B, 1, 1)
        attn_output, _ = self.cross_attn(query=queries, key=context_seq, value=context_seq)
        mode_features = self.layer_norm(queries + attn_output)

        # --------------------------------------------------------
        # Step 4: 解码控制信号 (Action Generation)
        # --------------------------------------------------------
        # 输入: [B, num_modes, hidden_dim]
        # 输出: [B, num_modes, steps*2]
        flat_actions = self.control_head(mode_features)
        
        # Reshape: [B, Modes, Time, 2]
        actions = flat_actions.view(B, self.num_modes, self._future_steps, 2)
        plan_trajs = torch.stack([bicycle_model(actions[:, i], ego[:, -1])[:, :, :3] for i in range(agent2map_modal*agent2agent_modal)], dim=1)

        # Step 5: 解码分数 (Scoring)
        plan_trajs_input = plan_trajs.view(B, self.num_modes, -1) # Flatten
        traj_embed = self.traj_encoder(plan_trajs_input) # [B, Modes, 64]
        global_context = context_seq.view(B, -1) # [B, hidden_dim*4]
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.num_modes, -1) #[B, hideden_dim*4] -> [B, Modes, hidden_dim*4]
        score_input = torch.cat([
            mode_features,  # 意图 (Intent) [B, Modes, hidden_dim]
            traj_embed,     # 物理结果 (Physics Result) [B, Modes, 64]
            global_context_expanded  # 环境约束 (Map/Obstacles) [B, Modes, hidden_dim*4]
        ], dim=-1)
        
        scores = self.score_head(score_input).squeeze(-1) # [B, Modes]

        # Step 6: 生成 Cost Weights 
        raw_cost_weights = self.cost_head(global_context)
        cost_function_weights = torch.cat(
            [raw_cost_weights[:, :7] * self.scale, self.constraint.expand(B, -1)], 
            dim=-1
        )

        return actions, scores, cost_function_weights

# class AVDecoder(nn.Module):
#     def __init__(self, future_steps=50, feature_len=9, num_modes=agent2map_modal*agent2agent_modal, hidden_dim=256):
#         """
#         num_modes: 规划出的轨迹模态数量 (例如 3: 左换道, 保持, 右换道)
#         """
#         super(AVDecoder, self).__init__()
#         self._future_steps = future_steps
#         self.num_modes = num_modes 
        
#         # 1. 共享特征提取网络 (Backbone)
#         # 将融合后的特征映射到高维潜在空间
#         self.shared_net = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim * 3, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU()
#         )

#         # 2. 多模态控制头 (Multi-modal Control Head)
#         # 一次性输出所有模态的控制序列
#         # 维度: [num_modes * future_steps * 2] -> 2代表(ax, ay) 或 (x, y)
#         self.control_head = nn.Linear(256, num_modes * future_steps * 2)
        
#         # 3. 规划打分头 (Scoring Head)
#         # 预测哪条规划轨迹是最好的 (Logits)
#         self.score_head = nn.Sequential(
#             nn.Linear(256 * 2, 128),
#             nn.ELU(),
#             nn.Linear(128, num_modes) 
#         )

#         # 4. 成本函数权重头 (Cost Function Weights Head)
#         # 这部分保持基于场景生成，不需要多模态，因为物理约束对所有轨迹通用
#         self.cost_head = nn.Sequential(
#             nn.Linear(hidden_dim * 4, 128),
#             nn.ReLU(),
#             nn.Linear(128, feature_len),
#             nn.Softmax(dim=-1)
#         )
        
#         # 注册缓冲区 (不需要梯度)
#         self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
#         self.register_buffer('constraint', torch.tensor([[10, 10]]))


#     def forward(self, agent_map, agent_agent, ego_actor, map_feature):
#         # 1. 特征融合
#         # 去掉多余的维度，拼接 Map 和 Agent 特征
#         #ipdb.set_trace()
#         combined = torch.cat([agent_map, agent_agent, ego_actor], dim=-1) # [B, 3*D]
#         B = combined.shape[0]

#         # 2. 提取共享特征
#         feature = self.shared_net(combined) # [B, 256]

#         # 3. 生成多模态控制序列 (Actions)
#         flat_actions = self.control_head(feature)
#         # Reshape: [B, Modes, Time, 2]
#         actions = flat_actions.view(B, self.num_modes, self._future_steps, 2)

#         # 4. 生成每条轨迹的推荐分数 (Scores)
#         feature_detached = feature
#         score_input = torch.cat([feature_detached, map_feature], dim=-1)
#         scores = self.score_head(score_input) # [B, Modes]

#         # 5. 生成成本权重 (Cost Weights)
#         cost_input = torch.cat([combined, map_feature], dim=-1) # [B, 256*4]
#         raw_cost_weights = self.cost_head(cost_input)
#         cost_function_weights = torch.cat(
#             [raw_cost_weights[:, :7] * self.scale, self.constraint.expand(B, -1)], 
#             dim=-1
#         )

#         return actions, scores, cost_function_weights


# Build predictor
class Predictor(nn.Module):
    def __init__(self, future_steps):
        super(Predictor, self).__init__()
        self._future_steps = future_steps

        # agent layer
        self.vehicle_net = AgentEncoder()
        self.pedestrian_net = AgentEncoder()
        self.cyclist_net = AgentEncoder()

        # map layer
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()
        
        # attention layers
        self.agent_feature = AgentFeature()
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        # decode layers
        self.plan = AVDecoder(self._future_steps)
        self.predict = AgentDecoder(self._future_steps)


    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        # actors
        ego_actor = self.vehicle_net(ego)
        vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1) 
        pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1) 
        cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]
        # raw_actors = torch.cat([ego.unsqueeze(1), neighbors[..., :8]], dim=1)
        # actors_history = self.actor_history_net(raw_actors)

        # maps
        lane_feature = self.lane_net(map_lanes)
        crosswalk_feature = self.crosswalk_net(map_crosswalks)
        lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
        crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
        map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
        map_mask[:, :, 0] = False # prevent nan 
        #ipdb.set_trace()
        
        

        agent_feature = self.agent_feature(actors, actor_mask)

        # map to actor
        map_feature = []
        agent_map = []
        #ipdb.set_trace()
        # 1. 获取维度信息
        B, A, D = agent_feature.shape  # [32, 11, 256]
        _, _, L_num, L_points, _ = lane_feature.shape # [32, 11, 6, 100, 256]
        _, _, C_num, C_points, _ = crosswalk_feature.shape # [32, 11, 4, 100, 256]

        # 2. 压扁 (Flatten) 输入：合并 B 和 A
        # [32, 11, 256] -> [352, 256]
        agent_flat = agent_feature.view(-1, D) 
        # [32, 11, 6, 100, 256] -> [352, 6, 100, 256]
        lane_flat = lane_feature.view(-1, L_num, L_points, D)
        # [32, 11, 4, 100, 256] -> [352, 4, 100, 256]
        cw_flat = crosswalk_feature.view(-1, C_num, C_points, D)
        # [32, 11, 10] -> [352, 10]
        mask_flat = map_mask.view(-1, map_mask.shape[-1])

        # 3. 一次性通过 Agent2Map
        # 输入: [352, ...] 
        map_feat_flat, agent_map_flat = self.agent_map(agent_flat, lane_flat, cw_flat, mask_flat)

        # 4. 还原 (Restore) 维度
        # [352, 256] -> [32, 11, 256]
        map_feature = map_feat_flat.view(B, A, -1)   
        agent_map = agent_map_flat.view(B, A, -1)
        #ipdb.set_trace()

        # for i in range(actors.shape[1]):
        #     output = self.agent_map(agent_feature[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i])
        #     map_feature.append(output[0])
        #     agent_map.append(output[1])
        # map_feature = torch.stack(map_feature, dim=1)
        # #ipdb.set_trace()
        # agent_map = torch.stack(agent_map, dim=1).squeeze(2) # shape [B, N, 256]
        # #ipdb.set_trace()

        # agent to agent
        agent_agent = self.agent_agent(agent_feature, actor_mask) # shape [B, N, 256]

        # plan + prediction 
        # plans, cost_function_weights = self.plan(agent_map[:, :, 0], agent_agent[:, :, 0], actors_history[:, 0])
        # predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, :, 1:], neighbors[:, :, -1], actors_history[:, 1:])
        #ipdb.set_trace()
        plans, plan_scores, cost_function_weights = self.plan(agent_map[:, 0], agent_agent[:, 0], ego_actor, ego,  map_feature[:, 0])
        #ipdb.set_trace()
        predictions, prediction_scores = self.predict(agent_map[:, 1:], agent_agent[:, 1:], neighbor_actors, neighbors[:, :, -1], map_feature[:, 1:])

        #plan_scores_old, prediction_scores_old = self.score(map_feature, agent_feature, agent_map, agent_agent, actor_mask)
        #ipdb.set_trace()
        
        return plans, predictions, plan_scores, prediction_scores, cost_function_weights, agent_map, agent_agent

if __name__ == "__main__":
    # set up model
    model = Predictor(50)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))
