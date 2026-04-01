1. the per head models sanity check still has a huge gap after 200 steps on real 2 node 16 gpus distributed runs compraing to the global moe and standard moe. can you check why, like run and figure out why the discrepancy. Like a few things to check, i want you to make a per step check:

a. have the same weights init as global moe
b. have the same data points and lr, seed and etc.
c. for every single step, check loss, weights difference delta, and optimizer states difference and etc.
d. ideally everything should be very similar, we have huge gap after 200 steps.


can we check this again, we did some check and fixed qk norm. but there are still discrepancy, we want to run for much longer like 200-500 steps to see clear difference why huge gaps between ce loss and just figure out if there are bugs in our per head model codes. run all sanity check and unit test possible. use gpu as you want on this machine, and use uv run and real data to test out, can also create fake synthetic data for unit tests as well.