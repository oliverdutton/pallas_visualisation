from collections import defaultdict, namedtuple
from dataclasses import dataclass

import numpy as np

from .data import Load, Tensor,Store

# Custom class to stay inkeeping with Triton-viz datatypes
@dataclass
class Dtype:
  primitive_bytewidth: int = 1
  @property
  def element_ty(self):
    return namedtuple('element_ty', ['primitive_bitwidth'])(self.primitive_bytewidth*8)
  
op_mapping = {
    'get': Load, 'masked_load': Load,
    'swap': Store, 'masked_swap': Store
}
def analyse_logs(logs, annotate_by_function_signature=True):
  '''
  Only use `annotate_by_function_signature` if your pallas kernel function contains a flat arg signature
  '''
  tensors = {var_count: (Tensor(0, Dtype(1), (1,)*len(aval.shape), aval.shape, aval.dtype.itemsize), 0) for var_count, aval in logs['var'].items()}

  if (annotate_by_function_signature):
    # var_count_to_name = {var_count: name for var_count, name in zip(tensors.keys(), list(signature(f.traced_kernel_function).parameters.keys()))}
    for (tensor,_), argname in zip(tensors.values(), logs['argnames']):
      tensor.name = argname

  op_records = defaultdict(lambda: [])
  for grid_index in logs['indexing']:
    for entry in logs['indexing'][grid_index]:
      op_type, var_count, indexs = entry
      # print(op_type, var_count_to_name[var_count], indexs)
      op = op_mapping[op_type](
        ptr=var_count, # Ref indexed in to
        shape=indexs.shape, # shape of indexs
        offsets=np.arange(np.prod(indexs.shape)),
        access_masks=(indexs>=0), # mask on indexs
        invalid_access_masks=(indexs==np.iinfo(np.dtype("int32")).min),
        original_offsets=indexs, # indexs in to Ref
        original_masks=(indexs>=0),
      )
      op_records[grid_index].append(op)

  # Memory access info
  n_byte_accessed = {
    Load: 0,
    Store: 0,
  }
  n_byte_accessed_including_masked = {
    Load: 0,
    Store: 0,
  }
  for op_record in op_records.values():
    for op in op_record:
      tensor = tensors[op.ptr][0]
      bytes_per_element = tensor.dtype.primitive_bytewidth * tensor.element_size
      n_byte_accessed[type(op)] += bytes_per_element * op.original_masks.sum()    
      n_byte_accessed_including_masked[type(op)] += bytes_per_element * np.prod(op.original_masks.shape)
      
  analysis_data = [
    ('Grid', logs['grid']),
    ('Load', n_byte_accessed[Load]),
    ('Store', n_byte_accessed[Store]), 
    ('Masked Load Ratio', f'{n_byte_accessed[Load] / n_byte_accessed_including_masked[Load]:.3f}'),
    ('Masked Store Ratio', f'{n_byte_accessed[Store] / n_byte_accessed_including_masked[Store]:.3f}'),
  ]
  return op_records, tensors, analysis_data
  