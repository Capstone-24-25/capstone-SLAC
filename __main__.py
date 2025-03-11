import os
import torch
from SLAC25.utils import *
from SLAC25.network import ModelWrapper
from SLAC25.models import * # import the model
from argparse import ArgumentParser


########## Start ##########
ap = ArgumentParser()
ap.add_argument("--testmode", action="store_true", help="Enable test mode with small dataset")
ap.add_argument("--nepoch", type=int, default=10)
ap.add_argument("--outdir", type=str, default=None)
ap.add_argument("--lr", type=float, default=0.001)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = ap.parse_args()

if args.testmode: # if testmode, no need to specify outdir in slurmlogs
    args.outdir = "./models"
    args.nepoch = 3
    args.verbose = True
else:
    if args.outdir is None:
        try:
            slurm_jid = os.environ['SLURM_JOB_ID']
            slurm_jname = os.environ['SLURM_JOB_NAME']
            username = os.environ['USER']
            args.outdir = f"/scratch/slac/models/{username}.{slurm_jname}.{slurm_jid}"
            os.makedirs(args.outdir, exist_ok=True)
        except KeyError:
            args.outdir = "./models"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model wrapper
model = ResNet
model_wrapper = ModelWrapper(model_class=model, num_classes=4, keep_prob=0.75, num_epochs=args.nepoch, verbose=args.verbose, testmode=args.testmode, outdir=args.outdir)

# enable testmode for smaller sample size
# enable verbose for detailed info
model_wrapper._prepareDataLoader(batch_size=args.batch_size, testmode=args.testmode)

# train the model
train_log = model_wrapper.train()

# visualize training performance
visualize_performance(train_log, args.outdir, "train_log.png")
