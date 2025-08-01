from .badnet import create_badnet_poisoned_set
from .wanet import create_wanet_poisoned_trainset, create_wanet_poisoned_testset, generate_wanet_grids, determine_input_dimensions


def create_poisoned_set(args, subset):
    if args.attack == 'badnet':
        if not 0.0 < args.poisoning_rate <= 1.0:
            raise ValueError("Poisoning rate must be between 0 and 1")
        return create_badnet_poisoned_set(trigger_size=args.trigger_size, poisoning_rate=args.poisoning_rate, target_label=args.target_label, subset=subset)
    elif args.attack == 'wanet':
        if not (0.0 < args.poisoning_rate <= 0.3 or args.poisoning_rate == 1.0):
            raise ValueError("Poisoning rate must be between 0 and 0.3 or 1.0 (just for testing)")
        if args.poisoning_rate == 1.0:
            return create_wanet_poisoned_testset(args, subset)
        else:
            return create_wanet_poisoned_trainset(args, subset)
    else:
        raise ValueError(f"Invalid attack type: {args.attack}")



def get_wanet_grids(args, subset):
    # get input height
    _, _, _, input_height = determine_input_dimensions(subset)
    noise_grid, identity_grid = generate_wanet_grids(k=args.k, input_height=input_height)
    return noise_grid, identity_grid

