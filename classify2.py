import torch
from torch.autograd import Variable

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding


def classify_video(video_dir, video_name, model, opt):
    assert opt.mode in ['score', 'feature']

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=False)

    video_outputs = []
    video_segments = []

    with torch.no_grad():
        for i, (inputs, segments) in enumerate(data_loader):

            inputs = Variable(inputs)

            outputs = model(inputs)

            video_outputs.append(outputs.cpu().data)
            video_segments.append(segments)

    if video_outputs:
        video_outputs = torch.cat(video_outputs)
        video_segments = torch.cat(video_segments)

    results = dict()
    results['video'] = video_name
    results['features'] = video_outputs
    results['clips'] = video_segments

    return results

