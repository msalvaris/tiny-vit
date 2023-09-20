import zipfile
import os
import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import ipdb
import fire

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from typing import Tuple


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label
    


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        print(f"image_height image_width {image_height}")
        print(f"patch_height patch_width {patch_height}")

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        print(f"patch_dim {patch_dim}")
        print(f"num_patches {num_patches}")
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        ipdb.set_trace()
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def _predict(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return pred_pixel_values, masked_patches, masked_indices

    def forward(self, img):
        pred_pixel_values, masked_patches, _ = self._predict(img)
  
        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss
    
    def predict_pixels(self, img):
        pred_pixel_values, _, masked_indices = self._predict(img)
        return pred_pixel_values, masked_indices


def prepare_data(data_dir:str):

    with zipfile.ZipFile(os.path.join(data_dir,'train.zip')) as train_zip:
        train_zip.extractall(data_dir)
        
    with zipfile.ZipFile(os.path.join(data_dir,'test.zip')) as test_zip:
        test_zip.extractall(data_dir)


def visualize(model_output, batch_imgs, mask_indices, patch_height=16, patch_width=16, number_per_row=4):
    """_summary_

    Args:
        model_output (_type_): Pytorch tensor B x P x V 
        batch_imgs (_type_): Pytorch tensor B x C x H x W
        mask_indices (_type_): Pytorch tensor B x N
        patch_height (int, optional): _description_. Defaults to 16.
        patch_width (int, optional): _description_. Defaults to 16.

    Returns:
        _type_: _description_
    """
    unpatched_output = rearrange(model_output, 
                       'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                       p1 = patch_height, 
                       p2 = patch_width, 
                       h = 14, 
                       w = 14)
    
    
    batch_range = torch.arange(batch_imgs.shape[0])[:, None]
    mask = torch.ones((batch_imgs.shape))
    mask_patch = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h=14,w=14,p1 = patch_height, p2 = patch_width)
    
    mask_patch[batch_range,mask_indices.detach().cpu().numpy()]=0.0
    back_to_mask = rearrange(
        mask_patch, 
        'b (h w) (p1 p2 c) -> b c (w p1) (h p2)', 
        p1 = patch_height, 
        p2 = patch_width, 
        w = 14, 
        h = 14)
    imgs_grid = make_grid(batch_imgs, nrow=number_per_row)
    unmpatched_grid = make_grid(unpatched_output, nrow=number_per_row)
    mask_grid = make_grid(back_to_mask, nrow=number_per_row)

    img_composite = Image.composite(to_pil_image(imgs_grid), to_pil_image(unmpatched_grid), to_pil_image(mask_grid).convert(mode="L"))
    images = to_pil_image(imgs_grid)
    patches = to_pil_image(unmpatched_grid)
    mask = to_pil_image(mask_grid)
    return images,patches,mask,img_composite


def save_visualisations(vis_tuple:Tuple[Image.Image], prefix=None, folder=None):
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
    else:
        folder="./"
    
    for idx, vis in enumerate(vis_tuple):
        vis.save(os.path.join(folder,f"{prefix}{idx}.png"))



def train_mae(data_dir:str):
    # Training settings
    batch_size = 256
    
    lr = 1.5e-4
    gamma = 0.7
    seed = 42
    device = 'cuda'
    savepath = "mae_trained_longer.pt"
    print(data_dir)
    print(os.path.join(data_dir,"train",'*.jpg'))
    train_list = glob.glob(os.path.join(data_dir,"train",'*.jpg'))
    test_list = glob.glob(os.path.join(data_dir, "test", '*.jpg'))
    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(test_list, transform=test_transforms)


    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=False)

    model = ViT(
        dim = 1024,
        image_size = 224,
        patch_size = 16,
        num_classes = 2,
        channels = 3,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
        
    ).to(device)

    mae = MAE(
        encoder = model,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 6       # anywhere from 1 to 8
    ).to(device)
    
    epochs = 160
    num_epochs = 150
    number_warmup_epochs = 10
    # optimizer
    optimizer = optim.AdamW(mae.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    
    train_scheduler = CosineAnnealingLR(optimizer, num_epochs)

    def warmup(current_step: int):
        return current_step/number_warmup_epochs
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)

    scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs])

    try:
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0

            for data, label in train_loader:
                data = data.to(device)
                label = label.to(device)
                
                loss = mae(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss / len(train_loader)
                batch_count+=1
                if batch_count%10 == 0:
                    print(
                        f"Batch : {batch_count} - loss : {epoch_loss:.4f} | lr : {scheduler.get_last_lr()}"
                    )
            scheduler.step()
            print(
                f"-- Epoch : {epoch+1} - loss : {epoch_loss:.4f} | lr : {scheduler.get_last_lr()}"
            )
            # Visualise last batch
            with torch.no_grad():
                for data, label in valid_loader:
                    data = data.to(device)
                    model_output, mask_indices = mae.predict_pixels(data)
                    vis_tuple = visualize(model_output, data, mask_indices, patch_height=16, patch_width=16, number_per_row=16)
                    save_visualisations(vis_tuple, prefix=f"Epoch_{epoch}_", folder="tiny_vit_longer")
                    break
                

           

    except KeyboardInterrupt:
        print("Interrupting training")


    print(f"Saving model to {savepath}")
    torch.save(
        {
            "mae": mae.state_dict(),
            "model": model.state_dict()
        },
        savepath
    )
    

if __name__=="__main__":
    fire.Fire({
        "prepare-data": prepare_data,
        "train-mae": train_mae
    })



