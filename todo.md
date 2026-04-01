1. the per head models sanity check still has a huge gap after 200 steps on real 2 node 16 gpus distributed runs compraing to the global moe and standard moe. can you check why, like run and figure out why the discrepancy. Like a few things to check, i want you to make a per step check:

a. have the same weights init as global moe
b. have the same data points and lr, seed and etc.
c. for every single step, check loss, weights difference delta, and optimizer states difference and etc.
d. ideally everything should be very similar, we have huge gap after 200 steps.


