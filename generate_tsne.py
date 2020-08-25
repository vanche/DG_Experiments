import os
import glob
import random
import itertools
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


iteration_list = [250]
perplexity_list = [3]
pca_dim_list = [25]
learning_rate_list = [10]
SEED=2020

datasets_path = {
    'BDD':'../bdd100k/seg/images',
    'cityscape':'../cityscape/leftImg8bit',
    'GTA':'../GTA/*_images'
}

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)


def sample_images(datasets_path, SEED=2020, n_sample=200):
    images_path = []
    images_label = []

    set_seed(SEED)
    for key, dataset_path in datasets_path.items():
        all_image_path = glob.glob(os.path.join(dataset_path, '**', '*.*'), recursive=True)
        image_path_sample = random.sample(all_image_path, n_sample)

        images_path.extend(image_path_sample)
        images_label.extend([key] * n_sample)

    ids = [i for i in range(n_sample * len(datasets_path.keys()))]
    random.shuffle(ids)
    images_label = [images_label[i] for i in ids]
    images_path = [images_path[i] for i in ids]
    return images_path, images_label


def read_image(image_path, mode, IN=False, gray_norm=False):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512))
    if mode == 'RGB' or mode == 'L':
        img = img.convert(mode)
        img = np.array(img).astype(np.float32) / 255.
        if IN:
            rgb_mean_std = ([np.mean(img[:,:,i]) for i in range(3)], [np.std(img[:,:,i]) for i in range(3)])
            img = (img - rgb_mean_std[0]) / rgb_mean_std[1]
    elif mode == 'Y' or mode == 'YUV':
        yuv = img.convert('YCbCr')
        yuv = np.array(yuv).astype(np.float32) / 255.
        if gray_norm:
            gray_mean_std = ([np.mean(yuv[:,:,0]), 0.5, 0.5], [np.std(yuv[:,:,0]), 0.5, 0.5])
        elif IN:
            gray_mean_std = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        if IN:
            yuv = (yuv - gray_mean_std[0]) / gray_mean_std[1]
        if mode == 'Y':
            img = yuv[0]
        else:
            img = yuv
    else:
        raise ValueError('This is not valid mode')
    return img


def generate_tsne(images_path, images_label, iteration_list, perplexitay_list, pca_dim_list, learning_rate_list,
                  output_path='./csv_data', modes=['Y', 'YUV'], IN=False, gray_norm=False):
    os.makedirs(output_path, exist_ok=True)

    for mode in modes:
        images = []
        for image_path in images_path:
            img = read_image(image_path, mode, IN, gray_norm)
            images.append(img)

        images = np.stack(images)
        images = images.reshape([images.shape[0], -1])

        hparam_list = list(itertools.product(iteration_list, perplexitay_list, pca_dim_list, learning_rate_list))
        for hparam in hparam_list:
            iteration, perplexity, pca_dim, learning_rate = hparam
            pca = PCA(n_components=min(images.shape[0], pca_dim))
            images_pca = pca.fit_transform(images)

            tsne = TSNE(
                n_components=3,
                n_iter=iteration,
                perplexity=perplexity,
                learning_rate=learning_rate,
                random_state=SEED
            )

            emb = tsne.fit_transform(images_pca)
            emb_df = pd.DataFrame(emb, columns=['x', 'y', 'z'])
            emb_df['img_path'] = images_path
            emb_df['img_label'] = images_label
            emb_df_basename = f'tsne_{iteration}_{perplexity}_{pca_dim}_{learning_rate}_{mode}_'
            emb_df_basename += f'IN={IN}.csv' if mode == 'RGB' else f'IN={IN}_gn={gray_norm}.csv'
            emb_df_path = os.path.join(output_path, emb_df_basename)
            emb_df.to_csv(emb_df_path, index=False)



if __name__ == '__main__':
    images_path, images_label = sample_images(datasets_path)
    generate_tsne(images_path, images_label, iteration_list, perplexity_list, pca_dim_list, learning_rate_list,
                  modes=['Y'], IN=False)
    generate_tsne(images_path, images_label, iteration_list, perplexity_list, pca_dim_list, learning_rate_list,
                  modes=['RGB'], IN=True)
    generate_tsne(images_path, images_label, iteration_list, perplexity_list, pca_dim_list, learning_rate_list,
                  modes=['RGB', 'Y'], IN=False)
    generate_tsne(images_path, images_label, iteration_list, perplexity_list, pca_dim_list, learning_rate_list,
                  modes=['Y'], IN=True, gray_norm=False)
    generate_tsne(images_path, images_label, iteration_list, perplexity_list, pca_dim_list, learning_rate_list,
                  modes=['Y'], IN=True, gray_norm=True)


