import torch
from margipose.models.margipose_model import MargiPoseModelFactory
from margipose.models import load_model
from margipose.data.get_dataset import get_dataset


model = load_model('./margipose/models/pretrained/margipose-mpi3d.pth')
model.eval()

dataset = get_dataset('mpi3d-test', model.data_specs, use_aug=False)

print('Use ground truth root joint depth? {}'.format(known_depth))
print('Number of joints in evaluation: {}'.format(len(included_joints)))

df = run_evaluation_3d(model, device, loader, included_joints, known_depth=known_depth,
                        print_progress=True)