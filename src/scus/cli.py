"""Command line entry points for model inference and shared-residual Align."""

import argparse

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description='scUS U/S distance and frozen-encoder Align')
    parser.add_argument('command', choices=['prepare', 'distance', 'align-fit', 'align-project'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == 'prepare':
        from .data.prepare import prepare_data
        out = prepare_data(cfg, args.tag)
    elif args.command == 'distance':
        from .zero_shot import encode
        out = encode(cfg, args.tag, device_name=args.device)
    elif args.command == 'align-fit':
        from .align import fit
        out = fit(cfg, args.tag, device_name=args.device)
    else:
        from .align import project
        out = project(cfg, args.tag, device_name=args.device)
    print(out)


if __name__ == '__main__':
    main()
