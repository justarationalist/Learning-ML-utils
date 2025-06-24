def decay_rate(decay, per):
    return decay ** (1/per)

def print_weight_counts(model):
    total = 0
    print("parameters {")
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
            print(f"  {name}: {param.numel()},")
    print("}")
    print(f"total: {total}")
