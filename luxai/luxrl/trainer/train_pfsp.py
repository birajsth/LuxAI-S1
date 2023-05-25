#!/usr/bin/env python
import sys
import os
import random
import time

import numpy as np
from pathlib import Path
import torch


import luxai
from ..config import get_config

from ..ma.player import MainPlayer as MP
from ..ma.payoff import Payoff

from ..runner.luxai_runner_pfsp import LuxAIRunner as Runner

"""Train script for LuxAI."""

def parse_args(args, parser):
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)

    # fmt: on
    return args


def main(args):
    parser = get_config()
    args = parse_args(args, parser)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"


    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(luxai.__file__)))[
                       0] + "/results") / args.algorithm_name
    log_dir = str(run_dir / 'logs')
    save_dir = str(run_dir / 'models')

    if not os.path.exists(run_dir):
        os.makedirs(str(run_dir))

    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # wandb
    run = None
    if args.use_wandb:
        import wandb
        if args.wandb_key:
             wandb.login(key=args.wandb_key)
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            id=args.id,
            resume="allow",
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    # seed
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    
   
    payoff = Payoff()
    playeragent = MP(None, payoff)
    
    config = {
        "all_args": args,
        "agent":playeragent,
        "payoff":payoff,
        "device": device,
        "log_dir":log_dir,
        "save_dir":save_dir,
        "run_name":run_name,
        "wandb_run":run,
    }


    runner = Runner(config)
    runner.run()

    # post process
    runner.envs.close()


    #runner.writer.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])