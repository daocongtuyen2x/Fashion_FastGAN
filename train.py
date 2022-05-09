import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

from models import weights_init, Discriminator, ADGen
from operation import Viton, copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
l1loss = nn.L1Loss()
#l2loss = nn.MSELoss()


#torch.backends.cudnn.benchmark = True
batch_size = 4


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_big, rec_part] = net(data, label, part=part)
        err_l = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean()
        err_p = percept( rec_big, F.interpolate(data, rec_big.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err = err_l+(2/batch_size)*err_p
        err.backward()
        return pred.mean().item(), [rec_big, rec_part]
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 1024
    nlr = 0.00005
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list1 = [
            transforms.Resize((int(im_size),int(im_size))),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans1 = transforms.Compose(transform_list1)
    transform_list2 = [
            transforms.Resize((int(im_size),int(im_size)))
        ]
    trans2 = transforms.Compose(transform_list2)

    
    dataset = Viton(root = 'Viton_final', transform1=trans1, transform2 = trans2)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))



    
    #from model_s import Generator, Discriminator
    netG = ADGen()
    netG.apply(weights_init)

    netD = Discriminator()
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    #fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
    #fix_inputs = next(dataloader)
    #fix_input_stack = fix_inputs[1]
    #fix_rest = fix_input_stack[2].to(device)
    #fix_stack = percept.model.net.module.encoder.enc(fix_input_stack)
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        inputs = next(dataloader)
        real_image, stack_input = inputs[0].to(device), inputs[1]
        current_batch_size = real_image.size(0)

        fake_images = netG(stack_input[0], stack_input[1])

        real_image = DiffAugment(real_image, policy=policy) #real_image 256
        fake_images = DiffAugment(fake_images, policy=policy) # list [fake 256, fake 128]
        
        ## 2. train Discriminator
        netD.zero_grad()

        err_dr_real, [rec_img_big, rec_img_part] = train_d(netD, real_image, label="real")
        err_dr_fake = train_d(netD, fake_images.detach(), label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        l1 = l1loss(real_image, fake_images)
        perc = percept( real_image, fake_images).sum()
        err_g = -pred_g.mean() + (2/batch_size)*l1 + (2/batch_size)*perc

        print('\n')
        print('Dis:', 'err_dr_real:', err_dr_real, 'err_dr_fake:', err_dr_fake)
        print('Gen:')
        print('pred_g: ', -pred_g.mean().item())
        print('l1_loss: ', (2/batch_size)*l1.item())
        print('perceptual:', (2/batch_size)*perc.item())
        print('err_g:', err_g.item())


        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr_real+err_dr_fake, -err_g.item()))

        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                gen_imgs = netG(inputs[1][0], inputs[1][1])
                #gen_imgs = torch.where(fix_mask_re==1, fix_rest, gen_imgs)
                vutils.save_image(gen_imgs.add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_big, rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=4, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')


    args = parser.parse_args()
    print(args)

    train(args)
