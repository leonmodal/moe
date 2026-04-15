First take a look and docs and the codebase. correct anything thats not correct in the docs.

Now the second thing is that i think now the codebase is really messy and not organized. I want you to really organize the code in a clean way and there is only one unified training function and stuff. We also want to stay away from speedrun model for training and now just train our normal models like moe and stuff.

the models we need are just 
standard llm
standard moe
gloabl moe
mixture of everything, include fully independent and precompute kv

and we want to stay away from the speedrun archs. Instead we will be using qwen 3 / llama 3.1 archs

a few other codebases to take a look at are:
Megatron-LM
modal-nmoe
nmoe

take a look at those moe training and see any tricks can help us to train better. include stuff like gemm to have faster throughput and etc.

When you finish make a git push please.