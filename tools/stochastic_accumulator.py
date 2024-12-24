# Authored by: https://github.com/lodestone-rock
# Source: https://github.com/lodestone-rock/compass_optimizer/blob/main/experimental/compass_experimental_sr_bf16.py

import torch

class StochasticAccumulator:
    """
    # init your model
    your_fancy_model = YourFancyModel(*your_model_args)

    # apply stochastic grad accumulator hooks
    StochasticAccumulator.assign_hooks(your_fancy_model)

    # training
    while True:
        loss = your_fancy_model.loss(*your_model_input)
        for _ in range(grad_accum_length):
            loss.backward()

        # apply grad buffer back
        StochasticAccumulator.reassign_grad_buffer(your_fancy_model)

        optimizer.step()
        optimizer.zero_grad()
    """

    @staticmethod
    def stochastic_grad_accum(p):
        # hack by adding attributes to "grad"
        if hasattr(p, "acc_grad") and p.dtype == torch.bfloat16 and p.grad is not None:
            acc_grad_fp32 = p.acc_grad.to(torch.float32, copy=True)
            # acc_grad_fp32 += fp_32_grad
            # upcast the gradient and then add it to p.grad
            acc_grad_fp32.add_(p.grad.to(torch.float32))
            copy_stochastic_(p.acc_grad, acc_grad_fp32)
            del acc_grad_fp32
            del p.grad
        elif hasattr(p, "acc_grad") and p.grad is not None:
            p.acc_grad.add_(p.grad)
            del p.grad
        elif p.grad is not None:
            p.acc_grad = p.grad.clone()
            del p.grad
            

    @staticmethod
    def reassign_grad_buffer(model):
        for p in model.parameters():
            if hasattr(p, "acc_grad"):
                p.grad = p.acc_grad
                del p.acc_grad

    @staticmethod
    def assign_hooks(model):
        hooks = []
        for p in model.parameters():
            #if p.requires_grad or p.grad is not None:
            hook = p.register_post_accumulate_grad_hook(
                StochasticAccumulator.stochastic_grad_accum
            )
            hooks.append(hook)
        return hooks

# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))