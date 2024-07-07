from functools import partial
from collections import defaultdict
from itertools import chain
from inspect import signature
import numpy as np

import jax
from jax import numpy as jnp, random, vmap, jit
from jax.experimental import pallas as pl
from jax.experimental import io_callback
from jax.core import Var, ShapedArray, eval_jaxpr
from jax._src.pallas.core import AbstractMemoryRef
from jax._src.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal
from jax._src.util import safe_map, safe_zip
from jax._src.pallas.primitives import load_p as load_primitive
from jax._src.state.types import ReadEffect

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def pallas_call(pallas_f, *args, interpret=True, **kwargs):
  '''
  A normal pallas call, but logs the original pallas kernel
  (used to extract function signature later)
  '''
  jax_f = pl.pallas_call(pallas_f, *args, **kwargs, interpret=interpret)
  jax_f.__setattr__('traced_kernel_function', pallas_f)
  jax_f.__setattr__('traced_kernel_function_args', args)
  jax_f.__setattr__('traced_kernel_function_kwargs', kwargs)
  return jax_f

def rewrite_to_indexing_log_eqn(refvar, idxvar, outvar, maskvar, eqn, var_to_index_mapping):
  '''
  Builds an indexing equivalent to the original op by building a `masked_load` jaxpr equation
  '''
  index_refvar = var_to_index_mapping[refvar]
  index_outvar = Var('', ShapedArray(outvar.aval.shape, jnp.int32))
  index_othervar = Literal(-1, ShapedArray((), jnp.int32)) if (maskvar is not None) else None
  args = (index_refvar, idxvar, maskvar, index_othervar)
  invars, args_tree = jax.tree.flatten(args)
  index_eqn = JaxprEqn(
    invars=invars,
    outvars=[index_outvar],
    primitive=load_primitive,
    params = {
      'args_tree': args_tree, 
      'cache_modifier': None, 
      'eviction_policy': None, 
      'is_volatile': False
    },
    effects = {ReadEffect(0)},
    source_info=eqn.source_info,
    ctx=eqn.ctx
  )
  return index_eqn, index_outvar

def log_indexing(f, args):
  # TODO: avoid re-tracing itself (avoids some weird side effects)
  # f = pallas_call(f.traced_kernel_function, *f.traced_kernel_function_args, **f.traced_kernel_function_kwargs)

  logs = {
    'indexing': defaultdict(lambda: []),
    'var': {},
    'grid': None, # is overwritten in rewrite
    'argnames': list(signature(f.traced_kernel_function).parameters.keys()),
  }

  def _log_indexing(indexing_outvar, invar, name, grid):
    def _callback_fn(x, grid_index=None):
      logs['indexing'][tuple(map(int, grid_index)) if grid!=() else grid].append((name, invar.count, x))
    def get_program_ids():
      _ = jnp.array([pl.program_id(i) for i in range(len(grid))])
    full_jaxpr = jax.make_jaxpr(
      pl.pallas_call(get_program_ids, 
                    grid=grid, 
                    interpret=True, 
                    out_shape=()
                    ))()
    # Ugly way of getting to the pallas_call jaxpr eqns
    program_id_eqns = full_jaxpr.eqns[0].params['jaxpr'].eqns[0].params['jaxpr'].eqns

    args = (indexing_outvar,) 
    if (grid!=()):
      # Log grid index
      args += (program_id_eqns[-1].outvars[0],)

    logging_eqn = jax.make_jaxpr(
      lambda *args: io_callback(
        _callback_fn,
        None, # return val
        *args) # args
    )(*(arg.aval for arg in args)).eqns[0]

    eqns = program_id_eqns + [
      logging_eqn.replace(invars=list(args))
    ]
    return eqns

  def generate_indexing_hook(eqn, var_to_index_mapping, grid, name):
    _rewrite_to_indexing_log_eqn = partial(rewrite_to_indexing_log_eqn, eqn=eqn, var_to_index_mapping=var_to_index_mapping)
    if (name=='masked_load'):
      refvar, idxvar, maskvar, othervar = jax.tree.unflatten(eqn.params['args_tree'], eqn.invars)
      outvar = eqn.outvars[0]
      index_eqn, index_outvar = _rewrite_to_indexing_log_eqn(refvar, idxvar, outvar, maskvar)
    elif (name=='get'):
      refvar, outvar = eqn.invars[0], eqn.outvars[0]
      idxvar = jax.tree.unflatten(eqn.params['tree'], eqn.invars[1:])
      index_eqn, index_outvar = _rewrite_to_indexing_log_eqn(refvar, idxvar, outvar, None)
    elif name=='masked_swap':
      refvar, idxvar, valvar, maskvar = jax.tree.unflatten(eqn.params['args_tree'], eqn.invars)
      index_eqn, index_outvar = _rewrite_to_indexing_log_eqn(refvar, idxvar, valvar, maskvar)
    elif (name=='swap'):
      refvar, valvar = eqn.invars[:2]
      idxvar = jax.tree.unflatten(eqn.params['tree'], eqn.invars[2:])
      index_eqn, index_outvar = _rewrite_to_indexing_log_eqn(refvar, idxvar, valvar, None)
    else:
      raise NotImplementedError
    return[
      index_eqn,
      *_log_indexing(index_outvar, refvar, name, grid=grid),
    ]

  def update_pljaxpr_eqns(eqns, var_to_index_mapping, grid):
    '''
    Iterates through equations, if an Op on MemRef is found, an op indexing
    into a index tracing MemRef is introduced
    '''
    indexing_primitive_names = {'get', 'masked_load', 'swap', 'masked_swap'}
    i = 0
    while (i<len(eqns)):
      eqn = eqns[i]
      if (eqn.primitive.name in indexing_primitive_names):
        hook_out_eqns = generate_indexing_hook(eqn, var_to_index_mapping, name=eqn.primitive.name, grid=grid)
        for new_eqn in hook_out_eqns[::-1]:
          eqns.insert(i+1, new_eqn)
        i+=len(hook_out_eqns)
      i+=1
    return eqns

  def rewrite_pallas_call(eqns):
    '''
    For each MemRef array, we insert an indexing version of the MemRef which is just
    an unravelled iota of the flattened array shape into the pallas_call signature

    For each op on an MemRef (load/store, masked/not masked) in the jaxpr we add a 
    masked_load equation and side-effect op which logs which indices were accessed
    by applying a (masked) load into the indexing version of the MemRef.

    Note, this does seem to also rewrite the underlying jaxpr
    '''
    pleqn = eqns[0]
    assert pleqn.primitive.name=='pallas_call'

    ### Outside, for the pallas_call JaxprEqn, we have in and outvars
    invars, outvars = pleqn.invars, pleqn.outvars
    vars = invars + outvars
    index_eqns = [jax.make_jaxpr(lambda: jnp.arange(np.prod(var.aval.shape)).reshape(var.aval.shape))().eqns for var in vars]
    index_vars = [eqns[-1].outvars[0] for eqns in index_eqns]

    ### Inside, for the Jaxpr, the outvars are concatted on to the invars and are now MemRef's
    p = pleqn.params
    plvars = p['jaxpr'].invars
    plindex_vars = [Var('', AbstractMemoryRef(ShapedArray(var.aval.shape, jnp.int32), None)) for var in plvars]
    plvar_to_index_mapping = dict(zip(plvars, plindex_vars))

    maps = p['grid_mapping'].block_mappings
    maps = maps[:len(invars)] + maps[:] +  maps[len(invars):]
    grid = p['grid_mapping'].grid
    
    # Caching side-effect
    for plvar, var in zip(plvars, vars):
      logs['var'][plvar.count] = var.aval
    logs['grid'] = grid

    pleqn = pleqn.replace(
      invars = invars + index_vars, # We inject the indexing equivalents into the invars so now its (*invars, *indexing_vars, *outvars) as jaxpr signature
      params={
        **pleqn.params,
        'jaxpr': p['jaxpr'].replace(
          invars=plvars[:len(invars)] + plindex_vars + plvars[len(invars):],
          eqns = update_pljaxpr_eqns(p['jaxpr'].eqns, plvar_to_index_mapping, grid=grid),
        ),
        'in_shapes': p['in_shapes'] + p['in_shapes'] + p['out_shapes'],
        'grid_mapping': p['grid_mapping'].replace(block_mappings=maps),
      }
    )
    eqns = list(chain.from_iterable(index_eqns))+[pleqn]
    return eqns

  def rewrite(obj):
    '''
    Recursive function that searches for the pallas_call primitive, 
    then rewrites it
    '''
    if isinstance(obj, ClosedJaxpr):
      return obj.replace(jaxpr=rewrite(obj.jaxpr))
    elif isinstance(obj, Jaxpr):
      eqns = obj.eqns
      if (eqns[0].primitive.name=='pallas_call'):
        eqns = rewrite_pallas_call(eqns)
      else:
        eqns = [rewrite(eqn) for eqn in eqns]
      return obj.replace(eqns=eqns)
    elif isinstance(obj, JaxprEqn):
      if hasattr(obj, 'params'):
        if 'jaxpr' in obj.params:
          return obj.replace(params={**obj.params,
                  'jaxpr': rewrite(obj.params['jaxpr'])})

  jaxpr = jax.make_jaxpr(f)(*args)
  index_jaxpr = rewrite(jaxpr)
  eval_jaxpr(index_jaxpr.jaxpr, index_jaxpr.consts, *args) # triggers side-effects to logs
  return logs
