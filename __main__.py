import os
import torch
from SLAC25.utils import *
from SLAC25.network import ModelWrapper
from SLAC25.models import * # import the model
from argparse import ArgumentParser


########## Start ##########
ap = ArgumentParser()
ap.add_argument("--testmode", action="store_true", help="Enable test mode with small dataset")
ap.add_argument("--phase1_nepoch", type=int, default=10, help="Number of epochs for phase 1 training")
ap.add_argument("--phase2_nepoch", type=int, default=10, help="Number of epochs for phase 2 training")
ap.add_argument("--outdir", type=str, default=None)
ap.add_argument("--lr", type=float, default=0.001)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--verbose", action="store_true", help="Enable verbose output")
ap.add_argument("--maxImgs", type=int, help="maximum number of images to train on")
ap.add_argument("--nwork", type=int, help="number of workers for loading images in parallel")
args = ap.parse_args()

if args.testmode: # if testmode, no need to specify outdir in slurmlogs
    args.outdir = "./models"
    args.phase1_nepoch = 2
    args.phase2_nepoch = 2
    args.verbose = True
else:
    if args.outdir is None:
        try:
            slurm_jid = os.environ['SLURM_JOB_ID']
            slurm_jname = os.environ['SLURM_JOB_NAME']
            username = os.environ['USER']
            args.outdir = f"/scratch/slac/models/{username}.{slurm_jname}.{slurm_jid}"
        except KeyError:
            args.outdir = "./models"

os.makedirs(args.outdir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model wrapper
model = ResNet(num_classes=4, keep_prob=0.75, hidden_num=512)
model_wrapper = ModelWrapper(
    model_class=model,
    num_classes=4,
    keep_prob=0.75,
    num_epochs=args.phase1_nepoch,  # Start with phase 1 epochs
    verbose=args.verbose,
    testmode=args.testmode,
    outdir=args.outdir
)

# enable testmode for smaller sample size
# enable verbose for detailed info
model_wrapper._prepareDataLoader(
    batch_size=args.batch_size, 
    testmode=args.testmode,
    max_imgs=args.maxImgs, 
    nwork=args.nwork)

# Phase 1 Training (224x224)
print("\n=== Starting Phase 1 Training (224x224) ===")
phase1_log, phase1_weights = model_wrapper.train_phase1()
visualize_performance(phase1_log, args.outdir, "train_log_phase1_224.png")

# Phase 2 Training (512x512)
print("\n=== Starting Phase 2 Training (512x512) ===")
model_wrapper.num_epochs = args.phase2_nepoch  # Update epochs for phase 2
phase2_log = model_wrapper.train_phase2(phase1_weights)
visualize_performance(phase2_log, args.outdir, "train_log_phase2_512.png")

# Save combined training logs
combined_log = {
    'phase1': phase1_log,
    'phase2': phase2_log
}

with open(f"{args.outdir}/combined_train_log.json", "w") as f:
    json.dump(combined_log, f, indent=4)