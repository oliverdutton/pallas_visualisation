# Pallas Visualisation

A tool to visualise the load/store ops in Pallas functions. Similar to [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) which the visualisation 

You can use any pallas machinery for writing your kernel, e.g. BlockSpecs or the grid, and visualise it.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oliverdutton/pallas_visualisation/blob/main/examples/demo.ipynb)

You can view indexing for a particular index on the grid   
<img src="https://github.com/oliverdutton/pallas_visualisation/blob/main/assets/grid_index_2.jpg" width="840" height="600">

or an overview of the whole grid, such as this add op split over four blocks

<img src="https://github.com/oliverdutton/pallas_visualisation/blob/main/assets/indexing_full_grid.jpg" width="840" height="600">

The tracing is done by adding MemRefs which track the indexing, on which all load/store ops are duplicated to operate on and then logged in an external callback by editing the [jaxpr](https://jax.readthedocs.io/en/latest/jaxpr.html).

Thanks to [Keren Zhou](https://www.jokeren.tech/) for [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz), from which the graphics display functions are taken.