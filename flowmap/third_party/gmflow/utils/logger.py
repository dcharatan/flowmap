import torch

from utils.flow_viz import flow_tensor_to_image


class Logger:
    def __init__(self, lr_scheduler,
                 summary_writer,
                 summary_freq=100,
                 start_step=0,
                 ):
        self.lr_scheduler = lr_scheduler
        self.total_steps = start_step
        self.running_loss = {}
        self.summary_writer = summary_writer
        self.summary_freq = summary_freq

    def print_training_status(self, mode='train'):

        print('step: %06d \t epe: %.3f' % (self.total_steps, self.running_loss['epe'] / self.summary_freq))

        for k in self.running_loss:
            self.summary_writer.add_scalar(mode + '/' + k,
                                           self.running_loss[k] / self.summary_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def lr_summary(self):
        lr = self.lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('lr', lr, self.total_steps)

    def add_image_summary(self, img1, img2, flow_preds, flow_gt, mode='train',
                          ):
        if self.total_steps % self.summary_freq == 0:
            img_concat = torch.cat((img1[0].detach().cpu(), img2[0].detach().cpu()), dim=-1)
            img_concat = img_concat.type(torch.uint8)  # convert to uint8 to visualize in tensorboard

            flow_pred = flow_tensor_to_image(flow_preds[-1][0])
            forward_flow_gt = flow_tensor_to_image(flow_gt[0])
            flow_concat = torch.cat((torch.from_numpy(flow_pred),
                                     torch.from_numpy(forward_flow_gt)), dim=-1)

            concat = torch.cat((img_concat, flow_concat), dim=-2)

            self.summary_writer.add_image(mode + '/img_pred_gt', concat, self.total_steps)

    def push(self, metrics, mode='train'):
        self.total_steps += 1

        self.lr_summary()

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.summary_freq == 0:
            self.print_training_status(mode)
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            tag = key.split('_')[0]
            tag = tag + '/' + key
            self.summary_writer.add_scalar(tag, results[key], self.total_steps)

    def close(self):
        self.summary_writer.close()
