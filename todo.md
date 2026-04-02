1. the per head models sanity check still has a huge gap after 200 steps on real 2 node 16 gpus distributed runs compraing to the global moe and standard moe. can you check why, like run and figure out why the discrepancy. Like a few things to check, i want you to make a per step check:

a. have the same weights init as global moe
b. have the same data points and lr, seed and etc.
c. for every single step, check loss, weights difference delta, and optimizer states difference and etc.
d. ideally everything should be very similar, we have huge gap after 200 steps.

can we check again, we run all here /tmp/moe/configs/depth_matched/4_layers but still mismatch. you can check my wandb and my modal workspace and see logs and to see the stuff. it was killed but you can see stopped app or just wandb and see there are differences. 

I need you to tell me why different, specifically global moe should be the same as per head sanity check, check if any bugs or sth and just fix.

you should be able to run code with gpu, use uv run and stuff. and we are training with 2 node 8 gpu ddp, but now you just have acess to 8 gpu, so just test the single node ddp to figure out all errs. like we basically need per head model sanity check to pass, in the sense that it should be similar to global moe.

potential stuff to check:

a. liger kernel
b. gradient checkpointing
c. gemm

basically figure out whats wrong, we need to have similar losses in long run before we can train the per head models successfully, as only then the results are meaningful.

once you finished, update status.md

I think the loss are still not reconciled. And there might be a few bugs in the other per head models '/tmp/moe/configs/depth_matched/4_layers' liek for example, for 4 layer models per head should be 8 because split into attention and mlp, am i right.

and check status.md we see that test on finewedu edu we just have huge differences. I need  you to again run ddp code and everything to lockin and fix those stuff. I need you to fix them, before returning to me the sanity check and the global moe should be logically the same and there shoudl not be a gap. if there is a gap then some code must be wrong.