1. 1 thing to fix is that i think now for all qkvo per head models, we are not doing batching which is not very efficient. 

2. i think dense attention is fine, my understanding is that for mlp branch, we could compute attention as well, but then just not using it, i.e. in the sense that not updating the kv. And then we can do flash attention or spda which actually resutls in faster throughput? One test to run is basically see which is faster, sparse attention or full. sparse meaning that for the branch that select mlp, we dont calcualte attention. instead we do gather and scatter to make everything faaster and efficient. i think this might depends on per layer how many percentage of tokens are using attention vs mlp, so a good thing to check.

3. i remeber previously we have some issue with precompute kv pre head. like my feeling is that we should jsut have 1 full table ish stuff per layer and then every single layer select from it correct? basically jsut correct anything thats not there and fix.

4. do we have the per layer version for block attention stuff? like not per head one? if not create

5. currently the logging is not correct. i want per step run time, liek sec/step and tokens/sec. dont do interpolation or average. i think slow down might be we have more active experts but verify

6. in general, use uv run and gpus to debug on 8 gpus ddp. and currently we are using accelerator. if you think write everything in torch is faster then change everything into torch native.

7. run unit test and make sure everything is correct. and we only need one of the md now we ahve arc md and mixture of everything md, just one but correct.

8. finally when all those down, we want to add a sanity check model. Specifically, for the per head precompute kv, we want a version where we alweays select attention at first layer and then mlp second layer and then etc. and for the attention head we want to always select the same haed for a layer so no mismathc. basically we want a hardcoded router decision for each attention layer. for mlp just do a bank and select. This sanity check is basically showing that this is bascialyl our global moe model so loss should somehow match.