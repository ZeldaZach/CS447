## Simple 3-Layer Neural Network

### How to use
1) `make`
2) `./cpu_neural train`
3) `./gpu_neural test [#Samples]`
    - Ex: `./gpu_neural test 20000`
    - The program will have an output file every 250 training nodes (250,500,...,60000)
    - Pre-compiled weight files are found in the `pre-compiled` directory. You can move these to `..` to enable loading of them into the program.

### Notes
Training: CUDA concurrency _not_ achieved for training due to race conditions within back_prop. System fails with batch loads due to global read/write updates.
Epochs are saved every 100 cycles, so you can run the tests against the file every so often

Testing: CUDA concurrency achieved for testing of the sample system. Since testing is a read-only operation, multiple threads/blocks can be used to simulate quick interactions with forward_learning.
