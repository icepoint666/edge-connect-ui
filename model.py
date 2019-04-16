import torch
import numpy as np

from networks import EdgeGenerator, InpaintGenerator

class Edgeconnect(object):
    def __init__(self, config):
        self.input_size = config.INPUT_SIZE
        self.batch_size = config.BATCH_SIZE

        self.edge_generator = EdgeGenerator(use_spectral_norm=True)
        self.inpainting_generator = InpaintGenerator()

    def load_model(self, config):
        ckpt_path = config.CKPT_DIR
        if ckpt_path:
            print('Model loaded from {}....start'.format(ckpt_path))
            edge_ckpt_path = ckpt_path + 'EdgeModel_gen.pth'
            if torch.cuda.is_available():
                data = torch.load(edge_ckpt_path)
            else:
                data = torch.load(edge_ckpt_path, map_location=lambda storage, loc: storage)

            self.edge_generator.load_state_dict(data['generator'])
            inpaint_ckpt_path = ckpt_path + 'InpaintingModel_gen.pth'
            if torch.cuda.is_available():
                data = torch.load(inpaint_ckpt_path)
            else:
                data = torch.load(inpaint_ckpt_path, map_location=lambda storage, loc: storage)

            self.inpainting_generator.load_state_dict(data['generator'])
            self.warmup(config)
            print('Model loaded from {}....end'.format(ckpt_path))
        else:
            print('Model loading is fail')

    def warmup(self, config):
        size = config.INPUT_SIZE
        bc = config.BATCH_SIZE
        images = torch.zeros((bc, 3, size, size))
        images_gray = torch.zeros((bc, 1, size, size))
        edges = torch.zeros((bc, 1, size, size))
        masks = torch.zeros((bc, 1, size, size))

        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_gray, edges_masked, masks), dim=1)
        outputs = self.edge_generator(inputs)
        edges = (outputs * masks + edges * (1 - masks)).detach()
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.inpainting_generator(inputs)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    def demo(self, config, images, images_gray, edges, masks):
        images = torch.from_numpy(images)
        images = images.permute(0,3,1,2)
        images_gray = torch.from_numpy(images_gray)
        images_gray = images_gray.unsqueeze(0).unsqueeze(0)
        edges = edges.astype('float64')
        edges = torch.from_numpy(edges)
        edges = edges.unsqueeze(0).unsqueeze(0)
        masks = masks.astype('float64')
        masks = torch.from_numpy(masks)
        masks = masks.permute(0,3,1,2)
        edges_masked = edges * (1.0 - masks)
        gray_masked = (images_gray * (1.0 - masks)) + masks
        images_masked = (images * (1.0 - masks)) + masks
        print(gray_masked.shape, edges_masked.shape, masks.shape)
        inputs = torch.cat((gray_masked, edges_masked, masks), dim=1).type(torch.FloatTensor)
        outputs = self.edge_generator(inputs)
        edges_1 = outputs.data.float() * masks.float()
        edges_2 = edges.float() * (1.0 - masks).float()
        edges = edges_1 + edges_2
        edges_res = edges.data.float().squeeze().numpy()
        inputs = torch.cat((images_masked.float(), edges.float()), dim=1)
        outputs = self.inpainting_generator(inputs)
        outputs_merged = (outputs.data.float() * masks.float()) + (images.float() * (1 - masks.float()))
        outputs_merged = outputs_merged.permute(0,2,3,1).numpy()
        return edges_res, outputs_merged
