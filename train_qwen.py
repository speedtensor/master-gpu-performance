import os
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from abc import ABC, abstractmethod
from huggingface_hub import login

# ==========================================
# 1. PROFILE CALLBACK INTERFACE & IMPLEMENTATIONS
# ==========================================

class ProfilerCallback(ABC):
    @abstractmethod
    def on_train_begin(self, args):
        pass

    @abstractmethod
    def on_step_begin(self, step: int):
        pass

    @abstractmethod
    def on_step_end(self, step: int, loss: float):
        pass

    @abstractmethod
    def on_train_end(self):
        pass


class PyTorchProfilerCallback(ProfilerCallback):
    def __init__(self, profile_start_step: int, profile_steps: int):
        self.profile_start_step = profile_start_step
        self.profile_steps = profile_steps
        self.profiler = None

    def on_train_begin(self, args):
        warmup_steps = 1 if self.profile_start_step >= 1 else 0
        wait_steps = max(0, self.profile_start_step - warmup_steps)
        
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=self.profile_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_profile_logs"),
            record_shapes=True,
            with_stack=True,
        )
        self.profiler.start()

    def on_step_begin(self, step: int):
        pass

    def on_step_end(self, step: int, loss: float):
        if self.profiler:
            self.profiler.step()

    def on_train_end(self):
        if self.profiler:
            self.profiler.stop()


class CudaProfilerApiCallback(ProfilerCallback):
    """
    Callback for Nsight Systems (nsys) or Nsight Compute (ncu) 
    triggered via CUDA Profiler API boundaries.
    """
    def __init__(self, profile_start_step: int, profile_steps: int):
        self.profile_start_step = profile_start_step
        self.end_step = profile_start_step + profile_steps

    def on_train_begin(self, args):
        pass

    def on_step_begin(self, step: int):
        if step == self.profile_start_step:
            print(f"--> [Step {step}] Starting CUDA Profiler API capture (nsys/ncu)...")
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()

    def on_step_end(self, step: int, loss: float):
        if step == self.end_step - 1:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"--> [Step {step}] Stopped CUDA Profiler API capture.")

    def on_train_end(self):
        pass


class CallbackHandler:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, callback: ProfilerCallback):
        self.callbacks.append(callback)

    def on_train_begin(self, args):
        for cb in self.callbacks:
            cb.on_train_begin(args)

    def on_step_begin(self, step):
        for cb in self.callbacks:
            cb.on_step_begin(step)

    def on_step_end(self, step, loss):
        for cb in self.callbacks:
            cb.on_step_end(step, loss)

    def on_train_end(self):
        for cb in self.callbacks:
            cb.on_train_end()


# ==========================================
# 2. MAIN EXECUTION SCRIPT
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2-VL Callback Profiling Script")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--total_steps", type=int, default=10, help="Total steps to run (safely increased)")
    parser.add_argument("--profile_start_step", type=int, default=3, help="Step to start profiling")
    parser.add_argument("--profile_steps", type=int, default=3, help="Number of steps to profile")
    parser.add_argument("--max_pixels", type=int, default=512 * 512, help="Max pixels for image tokens")
    parser.add_argument("--max_length", type=int, default=1024, help="Max text sequence length")
    parser.add_argument("--enable_torch_profiler", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--enable_cuda_profiler_api", action="store_true", help="Enable CUDA API triggers")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Authenticate with Hugging Face if token is provided via environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face using environment token...")
        login(token=hf_token)

    min_required_steps = args.profile_start_step + args.profile_steps + 2
    if args.total_steps < min_required_steps:
        args.total_steps = min_required_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Batch Size: {args.batch_size}")

    # Load Model & Processor on the fly from Hugging Face
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="sdpa"  # Explicitly set to avoid FlashAttention-2 errors
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    # Prepare Data Inputs on the fly
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
    single_message = {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image briefly."}]}
    messages_batch = [single_message for _ in range(args.batch_size)]

    texts = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_batch]
    
    image_inputs_list, video_inputs_list = [], []
    for msg in messages_batch:
        img_in, vid_in = process_vision_info([msg])
        if img_in: image_inputs_list.extend(img_in)
        if vid_in: video_inputs_list.extend(vid_in)

    inputs = processor(
        text=texts, images=image_inputs_list or None, videos=video_inputs_list or None,
        padding=True, max_length=args.max_length, truncation=False, return_tensors="pt"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Register Callbacks dynamically
    callback_handler = CallbackHandler()
    if args.enable_torch_profiler:
        callback_handler.add_callback(PyTorchProfilerCallback(args.profile_start_step, args.profile_steps))
    if args.enable_cuda_profiler_api:
        callback_handler.add_callback(CudaProfilerApiCallback(args.profile_start_step, args.profile_steps))

    # Training Loop with Granular NVTX Annotations
    print(f"Starting training loop ({args.total_steps} steps)...")
    callback_handler.on_train_begin(args)

    for step in range(args.total_steps):
        callback_handler.on_step_begin(step)
        
        # 1. Step Range
        torch.cuda.nvtx.range_push(f"Step_{step}")
        
        optimizer.zero_grad()
        
        # 2. Forward Pass
        torch.cuda.nvtx.range_push("Forward")
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        torch.cuda.nvtx.range_pop()
        
        # 3. Backward Pass
        torch.cuda.nvtx.range_push("Backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        
        # 4. Optimizer Step
        torch.cuda.nvtx.range_push("Optimizer_Step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_pop() # End Step_{step}
        
        print(f"Step {step}/{args.total_steps} - Loss: {loss.item():.4f}")
        callback_handler.on_step_end(step, loss.item())

    callback_handler.on_train_end()
    print("Training and profiling completed successfully.")

if __name__ == "__main__":
    main()