import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import math

from utils.serialization import load_adj

# Load adjacency and environmental matrices
ADJ_MX = torch.tensor(np.load('dataset/adjacency_matrix.npy'), dtype=torch.float32)
ENV_MX = torch.tensor(np.load('dataset/context_features.npy'), dtype=torch.float32)

def test_tensor_properties(tensor):
    """
    Print comprehensive tensor statistics for debugging purposes.
    
    Args:
        tensor (torch.Tensor): The tensor to analyze.
    """
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean():.6f}")
    print(f"Std: {tensor.std():.6f}")
    print(f"Min: {tensor.min():.6f}")
    print(f"Max: {tensor.max():.6f}")
    print(f"Contains NaN: {torch.isnan(tensor).any()}")
    print(f"Contains Inf: {torch.isinf(tensor).any()}")
    print(f"Requires grad: {tensor.requires_grad}")

def denormalize_data(normalized_data, means, stds, padding=0):
    """
    Restore original scale and range of normalized data.
    
    Args:
        normalized_data (torch.Tensor): Normalized data with shape [B, T, nvars].
        means (torch.Tensor): Means used for normalization.
        stds (torch.Tensor): Standard deviations used for normalization.
        padding (int, optional): Padding value for dimension expansion. Default is 0.
    
    Returns:
        torch.Tensor: De-normalized data with original scale and range.
    """
    # Restore original scale
    restored_scale = normalized_data * (stds.repeat(1, padding, 1))
    # Restore original range
    denormalized = restored_scale + (means.repeat(1, padding, 1))
    return denormalized

class BatchGraphConvolution(nn.Module):
    """
    Batch Graph Convolution layer for processing multiple graphs simultaneously.
    
    This layer performs graph convolution on batched input with corresponding
    adjacency matrices, enabling efficient processing of multiple graph instances.
    
    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
    """
    
    def __init__(self, in_features, out_features):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_features, adjacency_matrices):
        """
        Forward pass for batch graph convolution.
        
        Args:
            input_features (torch.Tensor): Input features [batch_size, N, in_features].
            adjacency_matrices (torch.Tensor): Adjacency matrices [batch_size, N, N].
        
        Returns:
            torch.Tensor: Output features [batch_size, N, out_features].
        """
        # Apply linear transformation to input features
        support = torch.einsum('bni,io->bno', input_features, self.weight)
        # Apply graph convolution with adjacency matrices
        output = torch.einsum('bnm,bmo->bno', adjacency_matrices, support)
        return output

class RecurrentCycle(nn.Module):
    """
    Temporal Periodicity Modeling for thermodynamic cycles with weekly variation adjustment.
    
    This module implements learnable daily cycles to explicitly model temporal 
    periodic patterns. It maintains learnable parameters Q ∈ R^(M×D) for M groups 
    with cycle length W=24. Weekly variation filtration is achieved by learning 
    daily cycle weights and biases conditioned on the day of the week.

    Args:
        cycle_len (int): Length of daily cycle, typically 24 for hourly data.
        channel_size (int, optional): Number of channels. Default is 1.
        week_dim (int, optional): Dimension of weekly embedding. Default is 16.
    """
    
    def __init__(self, cycle_len, channel_size=1, week_dim=16):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len  # W=24 in paper
        self.channel_size = channel_size
        self.week_dim = week_dim
        
        # Learnable daily cycle parameters
        self.daily_cycle = nn.Parameter(
            torch.zeros(cycle_len, channel_size), 
            requires_grad=True
        )
        
        # Weekly embedding for days of the week (0-6)
        self.day_embedding = nn.Embedding(
            num_embeddings=7, 
            embedding_dim=week_dim
        )
        
        # Linear layers to compute weights and biases from weekly embedding
        self.weight_projection = nn.Linear(week_dim, self.channel_size, bias=False)
        self.bias_projection = nn.Linear(week_dim, self.channel_size, bias=False)
        
    def forward(self, time_index, sequence_length, day_of_week_index):
        """
        Forward pass that adjusts daily cycles with weekly weights and biases.

        Args:
            time_index (torch.Tensor): Current time indices [batch_size].
            sequence_length (int): Length of sequence to generate.
            day_of_week_index (torch.Tensor): Day of the week indices [batch_size].

        Returns:
            torch.Tensor: Adjusted cycle data [batch_size, sequence_length, channel_size].
        """
        batch_size = time_index.size(0)
        device = time_index.device
        
        # Compute weekly embedding
        weekly_embedding = self.day_embedding(day_of_week_index)  # (batch_size, week_dim)
        
        # Generate weekly weights and biases
        weekly_weight = torch.relu(self.weight_projection(weekly_embedding))  # (batch_size, channel_size)
        weekly_bias = torch.relu(self.bias_projection(weekly_embedding))      # (batch_size, channel_size)
        
        # Align daily cycle data
        cycle_indices = (time_index.view(-1, 1) + torch.arange(sequence_length, device=device)) % self.cycle_len
        daily_cycle_data = self.daily_cycle[cycle_indices]  # (batch_size, sequence_length, channel_size)
        
        # Apply weekly adjustments to daily cycles
        adjusted_cycle_data = (daily_cycle_data * weekly_weight.unsqueeze(1) + 
                              weekly_bias.unsqueeze(1))  # (batch_size, sequence_length, channel_size)
        
        return adjusted_cycle_data

class Model(nn.Module):
    """
    Main DeepUHI model for fine-grained urban heat island effect forecasting.
    
    This model implements a context-aware thermodynamic modeling framework that
    combines temporal periodicity modeling with spatial graph convolution for
    accurate temperature prediction.
    
    Args:
        configs: Configuration object containing model parameters.
        dropout_rate (float, optional): Dropout rate. Default is 0.3.
        **kwargs: Additional arguments for backward compatibility.
    """
    
    def __init__(self, configs, dropout_rate=0.3, **kwargs):
        super(Model, self).__init__()
        
        # Extract configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.num_nodes = configs.enc_in

        # Load group configuration
        group_data = pd.read_csv(configs.group_file)
        group_ids = group_data['group_id'].values
        self.channel_to_group = torch.tensor(group_ids, dtype=torch.long)
        self.group_num = self.channel_to_group.max().item() + 1

        print(f"Initialized with {self.group_num} groups")

        # Initialize components
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm_const = 0.75

        # Create RecurrentCycle modules for each group
        self.group_cycle_modules = nn.ModuleList([
            RecurrentCycle(cycle_len=self.cycle_len, channel_size=1) 
            for _ in range(self.group_num)
        ])

        # Build model backbone based on type
        self._build_model_backbone()

        # Regression layers for final prediction
        self.regression_mlp = nn.Sequential(
            nn.Linear(self.pred_len * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.d_model, self.pred_len)
        )
        
        self.regression_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=self.pred_len * 2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=self.pred_len, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def _build_model_backbone(self):
        """Build the main model backbone based on model type."""
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )
        elif self.model_type == 'gcn':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            )
            self.gcn_layer = PeriodicDynamicGCN(self.d_model, self.d_model, self.num_nodes)
            self.final_layer = nn.Linear(self.d_model, self.pred_len)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _remove_periodic_component(self, input_data, cycle_index, day_index, device):
        """
        Remove periodic component from input data.
        
        Args:
            input_data (torch.Tensor): Input data [batch_size, seq_len, enc_in].
            cycle_index (torch.Tensor): Cycle indices [batch_size].
            day_index (torch.Tensor): Day indices [batch_size].
            device: Device for tensor operations.
            
        Returns:
            torch.Tensor: Data with periodic component removed.
        """
        batch_size, seq_len, enc_in = input_data.size()
        channel_to_group = self.channel_to_group.to(device)
        
        # Initialize periodic data tensor
        periodic_data = torch.zeros(batch_size, seq_len, enc_in, device=device)

        # Generate periodic data for each group
        for group_id in range(self.group_num):
            channel_indices = (channel_to_group == group_id).nonzero(as_tuple=True)[0]
            if channel_indices.numel() == 0:
                continue
                
            group_periodic = self.group_cycle_modules[group_id](cycle_index, seq_len, day_index)
            group_periodic = group_periodic.expand(-1, -1, channel_indices.size(0))
            periodic_data[:, :, channel_indices] = group_periodic

        return input_data - periodic_data

    def _add_periodic_component(self, prediction_data, cycle_index, day_index, device):
        """
        Add periodic component to prediction data.
        
        Args:
            prediction_data (torch.Tensor): Prediction data [batch_size, pred_len, enc_in].
            cycle_index (torch.Tensor): Cycle indices [batch_size].
            day_index (torch.Tensor): Day indices [batch_size].
            device: Device for tensor operations.
            
        Returns:
            torch.Tensor: Data with periodic component added.
        """
        batch_size, pred_len, enc_in = prediction_data.size()
        channel_to_group = self.channel_to_group.to(device)
        
        # Calculate future cycle and day indices
        future_cycle_index = (cycle_index + self.seq_len) % self.cycle_len
        future_day_index = (day_index + self.seq_len // 24) % 7
        
        # Initialize periodic data tensor
        periodic_data = torch.zeros(batch_size, pred_len, enc_in, device=device)

        # Generate periodic data for each group
        for group_id in range(self.group_num):
            channel_indices = (channel_to_group == group_id).nonzero(as_tuple=True)[0]
            if channel_indices.numel() == 0:
                continue
                
            group_periodic = self.group_cycle_modules[group_id](future_cycle_index, pred_len, future_day_index)
            group_periodic = group_periodic.expand(-1, -1, channel_indices.size(0))
            periodic_data[:, :, channel_indices] = group_periodic

        return prediction_data + periodic_data

    def forward(self, input_data, cycle_index, day_index):
        """
        Forward pass of the DeepUHI model.
        
        Args:
            input_data (torch.Tensor): Input data [batch_size, seq_len, enc_in].
            cycle_index (torch.Tensor): Cycle indices [batch_size].
            day_index (torch.Tensor): Day indices [batch_size].
        
        Returns:
            torch.Tensor: Temperature predictions [batch_size, pred_len, enc_in].
        """
        batch_size, seq_len, enc_in = input_data.size()
        device = input_data.device

        # Apply reversible instance normalization if enabled
        if self.use_revin:
            seq_mean = input_data.mean(dim=1, keepdim=True)
            seq_var = input_data.var(dim=1, keepdim=True) + 1e-5
            input_data = self.norm_const * (input_data - seq_mean) / torch.sqrt(seq_var)

        # Remove periodic component from input
        deperiodic_data = self._remove_periodic_component(input_data, cycle_index, day_index, device)

        # Apply model-specific processing
        if self.model_type == 'gcn':
            x = deperiodic_data.permute(0, 2, 1)  # [batch_size, enc_in, seq_len]
            x = self.model(x)  # Apply initial transformation
            x = self.gcn_layer(x, cycle_index)  # Apply GCN with cycle information
            x = self.final_layer(x)  # Apply final transformation
            predictions = x.permute(0, 2, 1)  # Return to [batch_size, pred_len, enc_in]
        else:
            x = deperiodic_data.permute(0, 2, 1)  # Adjust input shape
            output = self.model(x)
            predictions = output.permute(0, 2, 1)

        # Add periodic component to predictions
        final_predictions = self._add_periodic_component(predictions, cycle_index, day_index, device)

        # Reverse normalization if applied
        if self.use_revin:
            final_predictions = final_predictions * torch.sqrt(seq_var) + seq_mean

        return final_predictions

class PeriodicDynamicGCN(nn.Module):
    """
    Periodic Dynamic Graph Convolutional Network for thermal flow modeling.
    
    This module implements the periodic dynamic graph learning framework described 
    in the paper, which models thermal flow with time-varying adjacency matrices 
    for 24 daily periods.
    
    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        num_nodes (int): Number of nodes in the graph.
        dropout (float, optional): Dropout rate. Default: 0.3.
        hidden_dim (int, optional): Hidden dimension for node embeddings. Default: 64.
    """
    
    def __init__(self, in_features, out_features, num_nodes, dropout=0.3, hidden_dim=64):
        super(PeriodicDynamicGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_periods = 24  # Daily periods
        
        # Register static graph structure and environmental features
        self.register_buffer('static_adjacency', ADJ_MX.to_sparse().float())
        self.register_buffer('environmental_features', ENV_MX.float())
        self.env_dim = ENV_MX.shape[1]
        
        # Graph convolution parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # Periodic node embedding parameters
        self.node_bias = nn.Parameter(torch.zeros(self.num_nodes, self.hidden_dim))
        self.periodic_embeddings = nn.ParameterDict({
            'source_embeddings': nn.Parameter(torch.empty(self.num_periods, num_nodes, hidden_dim)),
            'target_embeddings': nn.Parameter(torch.empty(self.num_periods, num_nodes, hidden_dim))
        })
        
        # Environmental feature encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(self.env_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.residual_mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Activation and regularization
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Initialize parameters
        self._reset_parameters()
        
        # Cache for adjacency matrices
        self.adjacency_cache = {}

    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        for param in self.periodic_embeddings.values():
            nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def _compute_adaptive_adjacency(self, period_idx, env_embedding):
        """
        Compute adaptive adjacency matrix for a specific time period.
        
        Args:
            period_idx (int): Time period index [0-23].
            env_embedding (torch.Tensor): Encoded environmental features [N, hidden_dim].
            
        Returns:
            torch.Tensor: Adaptive adjacency matrix [N, N].
        """
        # Get period-specific node embeddings
        source_embeddings = (self.periodic_embeddings['source_embeddings'][period_idx].float() + 
                           env_embedding)
        target_embeddings = (self.periodic_embeddings['target_embeddings'][period_idx].float() + 
                           env_embedding)
        
        # Compute attention-based adjacency matrix
        attention_scores = torch.mm(source_embeddings, target_embeddings.t())
        adaptive_adjacency = self.activation(attention_scores)
        
        # Apply static topology constraints
        static_adj_dense = self.static_adjacency.to_dense().float()
        constrained_adjacency = adaptive_adjacency * static_adj_dense
        
        # Convert to sparse format and cache
        sparse_adjacency = constrained_adjacency.to_sparse().float()
        self.adjacency_cache[period_idx.item()] = sparse_adjacency

        return sparse_adjacency

    def forward(self, input_features, cycle_indices):
        """
        Forward pass with periodic dynamic graph convolution.
        
        Args:
            input_features (torch.Tensor): Input features [batch_size, num_nodes, in_features].
            cycle_indices (torch.Tensor): Cycle indices for each sample [batch_size].
            
        Returns:
            torch.Tensor: Output features [batch_size, num_nodes, out_features].
        """
        input_features = input_features.float()
        batch_size = input_features.shape[0]
        device = input_features.device
        
        # Encode environmental features
        env_embedding = self.env_encoder(self.environmental_features.to(device).float())
        
        # Get unique periods and process each separately
        period_indices = (cycle_indices % self.num_periods).long()
        unique_periods = torch.unique(period_indices)
        
        # Initialize output tensor
        output_features = torch.zeros_like(input_features, device=device).float()
        
        # Process each unique period
        for period_idx in unique_periods:
            # Get samples for this period
            batch_mask = (period_indices == period_idx)
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]
            
            # Compute period-specific adjacency matrix
            period_adjacency = self._compute_adaptive_adjacency(period_idx, env_embedding)
            
            # Extract features for current period samples
            period_features = input_features[batch_indices]  # [G, N, in_features]
            
            # Apply graph convolution
            support = torch.einsum('bnd,dm->bnm', period_features, self.weight.float())
            
            if period_adjacency.is_sparse:
                adj_dense = period_adjacency.to_dense()
                adj_batched = adj_dense.unsqueeze(0).repeat(support.shape[0], 1, 1)
                convolved = torch.bmm(adj_batched, support) + self.bias
            else:
                adj_batched = period_adjacency.unsqueeze(0).repeat(support.shape[0], 1, 1)
                convolved = torch.bmm(adj_batched, support) + self.bias
            
            # Store results
            output_features[batch_indices] = convolved
        
        # Add residual connection
        residual_output = self.residual_mlp(input_features)
        final_output = output_features + residual_output
        
        return self.dropout(self.activation(final_output))

    def extra_repr(self):
        """Extra representation string for debugging."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'num_nodes={self.num_nodes}, periods={self.num_periods}')
