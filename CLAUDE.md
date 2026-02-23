right now I am trying to do this:

Give an moe: 16 layers, 128 expert, 8 active, in total 2048 experts 


we want: Global mixture of experts, which is basically, each layer pick from the same 2048 pool of expert.

I want to run a pretraining experiments on this. We will have a normal moe training and a global mixture of experts training. lets use fsdp or ddp first but do create the code so we can run stuff. lets use accelerator. 

first tell me if you understand, if so give code. right now its a single node 8 gpu, and we will using stateful parquet dataset. you will have acess to the gpu. for env, lets create with uv

