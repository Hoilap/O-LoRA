# O-LoRA

- This repo releases our implementation for the O-LoRA model.
- It is built based on the pretrained T5-large model, and finetuned on our data.

![image_text](https://github.com/cmnfriend/O-LoRA/blob/main/data/O-LoRA.jpg)


## Setup

You can install the required libraries by running 

```
pip install -r requirements.txt
```

You are also required to download the t5-large model from huggingface, put it to the folder named ```initial_model```, and rename the model folder as 't5-large'.

LLaMA2 HF is also supported. You can put your llama2 hf model to the folder named ```initial_model``` and rename the model folder as 'llama'.


## Training and Evaluation

For t5-large:

You can reproduce our experiments of order 1 & 2 & 3 by simply running

order1:

```
bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts/order_2.sh> logs_and_outputs/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts/order_3.sh> logs_and_outputs/order_3/logs/train_and_infer.log 2>&1 &
```

The model you have trained will be saved in ```logs_and_outputs/order_1(2 or 3)/outputs```.

The result of each task will be saved in ```logs_and_outputs/order_1(2 or 3)/outputs/TASK_NAME/predict_results.json```.

You can also check the logs during training and infering in  ```logs_and_outputs/order_1(2 or 3)/logs/train_and_infer.log```

For LLaMA2:

order1:

```
bash scripts_llama/order_1.sh> logs_and_outputs_llama/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts_llama/order_2.sh> logs_and_outputs_llama/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts_llama/order_3.sh> logs_and_outputs_llama/order_3/logs/train_and_infer.log 2>&1 &
```
## Timestamp
O-lora复现成功，单卡

---

`ProcessGroupNCCL.cpp:4561 ... using GPU 0 to perform barrier as devices used by this process are currently unknown`。
*   **现象**：程序弹出警告，甚至可能导致**进程卡死（Hang/死锁）**。
*   **出现时机**：通常发生在分布式训练（DDP 或 DeepSpeed）的早期阶段，具体是在调用 `training_args.main_process_first` 上下文管理器时。
*   **根本原因**：在分布式环境（NCCL 进程组）尚未完全建立、或者当前进程尚未明确绑定到具体 GPU 设备时，程序就调用了同步操作（`barrier()`）。系统因为不知道当前进程该用哪个 GPU，所以默认尝试用 GPU 0，导致冲突或混乱。
* **解决方案** 
  *   **环境变量**：确认 DeepSpeed 或 torchrun 是否正确传递了 `LOCAL_RANK` 环境变量。
  *   **参数传递**：如果自动检测失效，需要在初始化 `TrainingArguments` 时显式传入 `local_rank` 参数。
---
```
[rank0]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/dengkn/O-LoRA/src/run_uie_lora.py", line 599, in <module>
[rank1]:     main()
[rank1]:   File "/home/dengkn/O-LoRA/src/run_uie_lora.py", line 539, in main
[rank1]:     train_result = trainer.train(resume_from_checkpoint=checkpoint)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/trainer.py", line 1591, in train
[rank1]:     return inner_training_loop(
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/trainer.py", line 1891, in _inner_training_loop
[rank1]:     with self.accelerator.accumulate(model):
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 119, in enter
[rank1]:     return next(self.gen)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py", line 1057, in accumulate
[rank1]:     cm_stack.enter_context(contextlib.nullcontext() if self.sync_gradients else self.no_sync(m))
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 448, in enter_context
[rank1]:     result = _cm_type.enter(cm)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 119, in enter
[rank1]:     return next(self.gen)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py", line 947, in no_sync
[rank1]:     with context():
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 119, in enter
[rank1]:     return next(self.gen)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 2397, in no_sync
[rank1]:     assert not self.zero_optimization_partition_gradients(), 
[rank1]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2
```

简单来说： 你在训练配置中开启了 DeepSpeed ZeRO Stage 2，同时又开启了 梯度累积（Gradient Accumulation）。
DeepSpeed ZeRO Stage 2：会对梯度（Gradients）进行切分（Partitioning），分布在不同的 GPU 上以节省显存。
no_sync：这是 PyTorch DDP 的一种机制，用于在梯度累积的前几步暂停多卡间的梯度同步，以减少通信开销。
冲突点：ZeRO Stage 2 要求梯度必须是切分状态，而 no_sync 上下文管理器试图以一种 ZeRO 2 不支持的方式处理梯度。DeepSpeed 为了保证数据正确性，显式地禁止了在 ZeRO 2 下使用 no_sync，从而抛出了这个 AssertionError。


accelerate               0.30.0
deepspeed                0.18.3
transformers             4.34.0

下载了指定deepspeed后解决

---

```
+ deepspeed --master_port 27244 src/run_uie_lora.py --do_train --do_predict --predict_with_generate --model_name_or_path /home/dengkn/N-LoRA/initial_model/llama --data_dir CL_Benchmark --task_config_dir configs/order1_configs/dbpedia --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir logs_and_outputs_llama/order_1/outputs/1-dbpedia --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-03 --num_train_epochs 1 --deepspeed configs/ds_configs/stage2_llama.config --run_name order1_round1 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name True --add_dataset_name True --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 10 --evaluation_strategy no --save_strategy no --save_steps 1500 --lamda_1 0.5 --lamda_2 0


-----Gradient checkpointing: False -----
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py:446: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
dataloader_config = DataLoaderConfiguration(dispatch_batches=None)
  warnings.warn(
Installed CUDA version 12.6 does not match the version torch was compiled with 12.4 but since the APIs are compatible, accepting this combination
Using /home/dengkn/.cache/torch_extensions/py39_cu124 as PyTorch extensions root...
Installed CUDA version 12.6 does not match the version torch was compiled with 12.4 but since the APIs are compatible, accepting this combination
Using /home/dengkn/.cache/torch_extensions/py39_cu124 as PyTorch extensions root...
Emitting ninja build file /home/dengkn/.cache/torch_extensions/py39_cu124/cpu_adam/build.ninja...
Building extension module cpu_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF cpu_adam.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/includes -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/TH -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/THC -isystem /home/dengkn/miniforge3/envs/aslora/include/python3.9 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/adam/cpu_adam.cpp -o cpu_adam.o 
[2/3] c++ -MMD -MF cpu_adam_impl.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/includes -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/TH -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/THC -isystem /home/dengkn/miniforge3/envs/aslora/include/python3.9 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/adam/cpu_adam_impl.cpp -o cpu_adam_impl.o 
[3/3] c++ cpu_adam.o cpu_adam_impl.o -shared -lcurand -L/usr/local/cuda/lib64 -L/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o cpu_adam.so
Loading extension module cpu_adam...
Time to load cpu_adam op: 22.258127450942993 seconds
Loading extension module cpu_adam...
Time to load cpu_adam op: 22.287405014038086 seconds
[rank0]:[E1218 20:27:21.426415540 ProcessGroupNCCL.cpp:629] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600081 milliseconds before timing out.
[rank1]:[E1218 20:27:21.433063423 ProcessGroupNCCL.cpp:629] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600088 milliseconds before timing out.
[rank1]:[E1218 20:27:21.484903450 ProcessGroupNCCL.cpp:2168] [PG ID 1 PG GUID 1 Rank 1]  failure detected by watchdog at work sequence id: 1 PG status: last enqueued work: 547, last completed work: -1
[rank0]:[E1218 20:27:21.484910731 ProcessGroupNCCL.cpp:2168] [PG ID 1 PG GUID 1 Rank 0]  failure detected by watchdog at work sequence id: 1 PG status: last enqueued work: 547, last completed work: -1
[rank1]:[E1218 20:27:21.484941868 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank0]:[E1218 20:27:21.484955368 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank1]:[E1218 20:27:21.484959605 ProcessGroupNCCL.cpp:681] [Rank 1] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E1218 20:27:21.484963240 ProcessGroupNCCL.cpp:681] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank1]:[E1218 20:27:21.484966856 ProcessGroupNCCL.cpp:695] [Rank 1] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E1218 20:27:21.484970331 ProcessGroupNCCL.cpp:695] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E1218 20:27:21.486305774 ProcessGroupNCCL.cpp:1895] [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600081 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7b99599e31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7b9906ffec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7b99070007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7b99070016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7b9959b425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7b995a69caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7b995a729c6c in /lib/x86_64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600081 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7b99599e31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7b9906ffec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7b99070007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7b99070016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7b9959b425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7b995a69caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7b995a729c6c in /lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7b99599e31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xe5c6fc (0x7b9906c5c6fc in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0x145c0 (0x7b9959b425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #3: <unknown function> + 0x9caa4 (0x7b995a69caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #4: <unknown function> + 0x129c6c (0x7b995a729c6c in /lib/x86_64-linux-gnu/libc.so.6)

  what():  [PG ID 1 PG GUID 1 Rank 1] Process group watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600088 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7ca805fe31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7ca7b35fec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7ca7b36007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7ca7b36016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7ca8061425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7ca806c9caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7ca806d29c6c in /lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7ca805fe31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xe5c6fc (0x7ca7b325c6fc in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0x145c0 (0x7ca8061425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #3: <unknown function> + 0x9caa4 (0x7ca806c9caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #4: <unknown function> + 0x129c6c (0x7ca806d29c6c in /lib/x86_64-linux-gnu/libc.so.6)

```
`ProcessGroupNCCL.cpp:629 [Rank 0] Watchdog caught collective operation timeout` NCCL 分布式同步超时 (Watchdog Timeout)
关闭offload
```
[rank0]:[E1218 20:43:21.166114205 ProcessGroupNCCL.cpp:629] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600036 milliseconds before timing out.
[rank0]:[E1218 20:43:21.166281407 ProcessGroupNCCL.cpp:2168] [PG ID 1 PG GUID 1 Rank 0]  failure detected by watchdog at work sequence id: 1 PG status: last enqueued work: 547, last completed work: -1
[rank0]:[E1218 20:43:21.166297611 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank0]:[E1218 20:43:21.166304091 ProcessGroupNCCL.cpp:681] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E1218 20:43:21.166309339 ProcessGroupNCCL.cpp:695] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E1218 20:43:21.167223203 ProcessGroupNCCL.cpp:1895] [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600036 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7564c37d71b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x756470dfec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x756470e007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x756470e016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7564c39365c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7564c449caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7564c4529c6c in /lib/x86_64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600036 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7564c37d71b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x756470dfec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x756470e007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x756470e016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7564c39365c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7564c449caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7564c4529c6c in /lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7564c37d71b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xe5c6fc (0x756470a5c6fc in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0x145c0 (0x7564c39365c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #3: <unknown function> + 0x9caa4 (0x7564c449caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #4: <unknown function> + 0x129c6c (0x7564c4529c6c in /lib/x86_64-linux-gnu/libc.so.6)

```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
双卡训练成功

---
推理出现问题。
改了一些错误的地方，问题解决。

---

最后会进行统一的推理，其实完全可以前三步不进行predict

--- 
其实具体的库版本配置应该看：https://github.com/BeyonderXX/InstructUIE

---
根据你提供的日志，train/learning_rate 显示为 0。这意味着在整个训练过程中，学习率实际上一直是 0，模型参数根本没有更新。因此，无论你将初始学习率设置为 1e-03 还是 1e-04，由于实际生效的学习率都是 0，训练出来的模型（以及最终的 Loss 和 predict_exact_match）自然是完全一样的。
造成这种情况的原因是 DeepSpeed 配置文件与训练参数冲突：
你的 DeepSpeed 配置文件 stage2_llama.config 中包含了一个 scheduler 部分（配置为 WarmupLR）。
当 DeepSpeed 配置文件中存在 scheduler 时，它会覆盖 HuggingFace Trainer 的调度器设置（即你脚本中的 --lr_scheduler_type constant 被忽略了）。
由于脚本中设置了 --warmup_steps 0，DeepSpeed 的 WarmupLR 调度器可能因此将学习率默认为了 0（或者使用了默认的 warmup_min_lr，通常为 0）。

有一丢丢提升

Global Batch Size（单卡训练和 8 卡训练的梯度累积必须对齐）。


---

删除了deepspeed的学习率调度，学习率在debuglog中能显示了
```
  0%|▌                                                                                                                 | 1/218 [01:03<3:51:27, 64.00s/it][LR Debug] global_step=1 current_lrs=[0.001] scheduled_lrs=[0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight norm=1.6333858966827393 updated=False grad_norm=None
[LR Debug] global_step=1 current_lrs=[0.001] scheduled_lrs=[0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight norm=1.6333858966827393 updated=False grad_norm=None
[LR Debug] global_step=2 current_lrs=[0.001] scheduled_lrs=[0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight norm=1  1%|█                                                                                                                 | 2/218 [02:07<3:49:59, 63.89s/it]
[LR Debug] global_step=2 current_lrs=[0.001] scheduled_lrs=[0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight norm=1.6333858966827393 updated=False grad_norm=None
[LR Debug] global_step=3 current_lrs=[0.001] scheduled_lrs=[0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight norm=1  1%|█▌                                                                                                                | 3/218 [03:09<3:44:59, 62.79s/it]
[LR Debug] global_step=3 current_lrs=[0.001] scheduled_lrs=[0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_A.default.weight norm=1.6333858966827393 updated=False grad_norm=None
```

改了学习率，使用断点进行调试，还把混合精度关掉了，无果
```
[LR Debug] global_step=3 current_lrs=[0.001] scheduled_lrs=[0.001, 0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_B.default.weight norm=18.1019344329834 updated=False grad_norm=None
[Debug] base_model.model.model.layers.0.self_attn.q_proj.loranew_B.default.weight grad: None
[LR Debug] global_step=4 current_lrs=[0.001] scheduled_lrs=[0.001, 0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_B.default.weight norm=18.1019344329834 updated=False grad_norm=None
  2%|█▊                                                                                                | 4/218 [03:34<3:11:21, 53.65s/it][Debug] base_model.model.model.layers.0.self_attn.q_proj.loranew_B.default.weight grad: None
[LR Debug] global_step=4 current_lrs=[0.001] scheduled_lrs=[0.001, 0.001] track=base_model.model.model.layers.0.self_attn.q_proj.loranew_B.default.weight norm=18.1019344329834 updated=False grad_norm=None
```

### 知识：什么是梯度累计
在深度学习训练中，Batch Size（批大小）对模型性能有很大影响：
训练稳定性： Batch Size 越大，计算出的梯度越接近整个数据集的真实梯度方向，训练过程更平稳，不容易陷入局部抖动。
收敛速度： 较大的 Batch Size 通常允许使用更大的学习率，从而加快模型的收敛速度。
现在的模型（如 BERT, Llama, ViT 等）参数量巨大。当你尝试增大 Batch Size 时，GPU 显存会迅速耗尽（OOM - Out of Memory），因为显存不仅要存模型参数，还要存前向传播的中间激活值（Activations）。
如果你只有一张 24G 显存的显卡，可能 Batch Size 设为 4 就满了，但实验发现 Batch Size 设为 64 效果才最好。这时候梯度累积就派上用场了。
缺点/局限性：
时间换空间： 梯度累积并不能加快训练速度。跑完一个“有效 Batch”的时间依然是 16 次小 Batch 计算的总和。
Batch Normalization (BN) 的坑： 这是一个常见误区。梯度累积不能解决 BN 在小 Batch 上的不准确问题。因为 BN 是在每一层前向传播时根据当前 Batch 计算均值和方差的。如果你单次输入的 Batch 只有 2，BN 还是会基于 2 来计算，梯度累积救不了它。（解决方法通常是改用 Layer Norm 或 Group Norm）。
### 各种step的定义
tqdm 步数：在 Trainer 中，进度条只有在 global_step 增加时才会跳动（即完成了一个完整的梯度累积周期，执行了 optimizer.step()）。
on_step_end 调用：这个回调函数在每一个 batch 处理完之后都会被调用。如果你设置 gradient_accumulation_steps=32，那么在 tqdm 步数增加 1 之前，on_step_end 已经跑了 32 次。

### 思考1
对aslora的的激活值进行层相似度计算，batchsize=1，怎么计算？好像也能计算，不过不同样本的稀疏度就不一样了。

## Citation
```latex
@article{wang2023orthogonal,
  title={Orthogonal Subspace Learning for Language Model Continual Learning},
  author={Wang, Xiao and Chen, Tianze and Ge, Qiming and Xia, Han and Bao, Rong and Zheng, Rui and Zhang, Qi and Gui, Tao and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2310.14152},
  year={2023}
}
```


