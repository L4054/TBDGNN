import torch
import torch.nn as nn
import numpy as np


class TimeCode(torch.nn.Module):
  def __init__(self, dimension,timestamp):
    super(TimeCode, self).__init__()
    self.dimension = dimension
    block_size = len(timestamp) // dimension
    timestamp_blocks = []
    for i in range(dimension):
      timestamp_blocks.append(timestamp[i*block_size])
    timestamp_blocks = np.array(timestamp_blocks)
    self.gap = torch.from_numpy(timestamp_blocks)


  def forward(self, t):
    device = t.device
    gap = self.gap.to(device)
    x = t.shape
    t = t.unsqueeze(2).expand(x[0],x[1], gap.shape[0])
    output = torch.where(t > gap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

    return output


class ProcessData(nn.Module):
  def __init__(self, CutTime_dim):
    super(ProcessData, self).__init__()
    self.CutTime_dim = CutTime_dim
  def forward(self, t):
    time_size = len(t) // self.CutTime_dim
    timestamp_blocks = []
    for i in range(time_size):

      timestamp_block = t[(i) * self.CutTime_dim]
      timestamp_blocks.append(timestamp_block.repeat(self.CutTime_dim))
    merged_timestamp_blocks = np.concatenate(timestamp_blocks, axis=0)
    if len(t) != len(merged_timestamp_blocks):
      cut_num = len(t) - len(merged_timestamp_blocks)
      if cut_num > 0:
        merged_timestamp_blocks = np.concatenate([merged_timestamp_blocks, t[-1].repeat(cut_num)], axis=0)
      else:
        merged_timestamp_blocks = merged_timestamp_blocks[:len(t)]
    return merged_timestamp_blocks.astype(np.float32)




