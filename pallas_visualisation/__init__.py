from .interface import launch
from .analysis import analyse_logs
from .tracer import pallas_call, log_indexing
visualise = lambda f, args, share=False, **kwargs: launch(*analyse_logs(log_indexing(f, args)), share=share, **kwargs)
