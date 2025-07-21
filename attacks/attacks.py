from badnet import create_badnet_poisoned_set
from wanet import create_wanet_poisoned_set


def create_poisoned_set(args, subset):
    if args.attack == 'badnet':
        if not 0.0 < args.poisoning_rate <= 1.0:
            raise ValueError("Poisoning rate must be between 0 and 1")
        return create_badnet_poisoned_set(trigger_size=args.trigger_size, poisoning_rate=args.poisoning_rate, target_label=args.target_label, subset=subset)
    elif args.attack == 'wanet':
        return create_wanet_poisoned_set(args, subset)
    else:
        raise ValueError(f"Invalid attack type: {args.attack}")
