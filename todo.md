1. the per head models sanity check still has a huge gap after 500 steps on real 2 node 16 gpus distributed runs compraing to the global moe and standard moe. can you check why, like run and figure out why the discrepancy

2. how can we have faster throughput in general, right now its slow, especially for the two per head models. Trying to make things faster. Any optimization we can do or sth?

3. right now we are not using any optimized kernel like gemm for multiple experts or sth. would be great if we can get something done there. Also we are not doing expert parallelization and etc. would be great to add those as well to boost through put. can check torch titan or megatron implementations. we need to focus on correctness and etc.

4. right now all wrapped in accelerator and etc. I am wondering if there is a much better way to write everything to be torch native so we can easily have control over stuff. should be a good optimization of code base to do. And we can then just use ddp or fsdp with good control and stuff that makes everything easy.

You should consult other code bases like clone megatron or torch titan or whatever and refer to those code for correctness.

So the two things are:

1. find ways to reconcile the discrepancy between per head sanity check model and global moe / standard moe, so we know that our per head code base is doing the correct thing. 

2. boost throughput through codebase optimization, new kernels, expert parallization and etc.