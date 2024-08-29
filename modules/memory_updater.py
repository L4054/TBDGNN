from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device
  #
  # def update_memory(self, unique_node_ids, unique_messages, timestamps):
  #   if len(unique_node_ids) <= 0:
  #     return
  #
  #   assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
  #                                                                                    "update memory to time in the past"
  #
  #   memory = self.memory.get_memory(unique_node_ids)
  #   self.memory.last_update[unique_node_ids] = timestamps
  #
  #   updated_memory = self.memory_updater(unique_messages, memory)
  #
  #   self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
# from model.new_timecode import TimeCode
# import torch
# from torch import nn
# from collections import defaultdict
#
# class Memory:
#     def __init__(self, n_nodes, memory_dimension, device):
#         self.memory = nn.Parameter(torch.zeros((n_nodes, memory_dimension + 1)).to(device), requires_grad=False)  # +1 to store timestamp
#         self.last_update = nn.Parameter(torch.zeros(n_nodes).to(device), requires_grad=False)
#         self.device = device
#
#     def get_memory(self, node_idxs):
#         return self.memory[node_idxs, :]
#
#     def get_last_update(self, node_idxs):
#         return self.last_update[node_idxs]
#
#     def set_memory(self, node_idxs, values):
#         self.memory[node_idxs, :] = values
#
#     def update_last_update(self, node_idxs, timestamps):
#         self.last_update[node_idxs] = timestamps
#
# class MemoryUpdater(nn.Module):
#     def update_memory(self, unique_node_ids, unique_messages, timestamps):
#         pass
#
# class SequenceMemoryUpdater(MemoryUpdater):
#     def __init__(self, memory, message_dimension, memory_dimension, device, time_dimension, timestamps):
#         super(SequenceMemoryUpdater, self).__init__()
#         self.memory = memory
#         self.layer_norm = nn.LayerNorm(memory_dimension)
#         self.message_dimension = message_dimension
#         self.device = device
#         self.time_encoder = TimeCode(time_dimension, timestamps).to(device)
#
#     def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
#         if len(unique_node_ids) <= 0:
#             return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
#
#         assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to update memory to time in the past"
#
#         current_memory = self.memory.get_memory(unique_node_ids)[:, :-1]  # Exclude timestamp from current memory
#         time_intervals = (timestamps - self.memory.get_last_update(unique_node_ids)).unsqueeze(1)
#         encoded_intervals = self.time_encoder(time_intervals)
#
#         updated_memory = self.memory_updater(unique_messages, current_memory)
#         updated_memory = torch.cat((updated_memory, encoded_intervals), dim=1)
#
#         updated_memory_data = self.memory.memory.data.clone()
#         updated_memory_data[unique_node_ids] = updated_memory
#
#         updated_last_update = self.memory.last_update.data.clone()
#         updated_last_update[unique_node_ids] = timestamps
#
#         return updated_memory_data, updated_last_update
#
# class GRUMemoryUpdater(SequenceMemoryUpdater):
#     def __init__(self, memory, message_dimension, memory_dimension, device, time_dimension, timestamps):
#         super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, time_dimension, timestamps)
#         self.memory_updater = nn.GRUCell(input_size=message_dimension, hidden_size=memory_dimension)
#
# class RNNMemoryUpdater(SequenceMemoryUpdater):
#     def __init__(self, memory, message_dimension, memory_dimension, device, time_dimension, timestamps):
#         super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, time_dimension, timestamps)
#         self.memory_updater = nn.RNNCell(input_size=message_dimension, hidden_size=memory_dimension)
#
# def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device, time_dimension, timestamps):
#     if module_type == "gru":
#         return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device, time_dimension, timestamps)
#     elif module_type == "rnn":
#         return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device, time_dimension, timestamps)
#
#
