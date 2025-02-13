import os
import random
import shutil

# 元のデータセットフォルダ
dataset_dir = '/Users/bruce.young/Downloads/colony_v23/images'
label_dir = '/Users/bruce.young/Downloads/colony_v23/labels'
all_images = os.listdir(dataset_dir)

# データをシャッフル
random.shuffle(all_images)

# 分割の割合
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 枚数の計算
total_images = len(all_images)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count

# フォルダの作成
output_dirs = ['train', 'valid', 'test']
for dir_name in output_dirs:
    os.makedirs(f'/Users/bruce.young/Downloads/colony_v23/{dir_name}/images', exist_ok=True)
    os.makedirs(f'/Users/bruce.young/Downloads/colony_v23/{dir_name}/labels', exist_ok=True)

# データの移動
for idx, image_file in enumerate(all_images):
    if idx < train_count:
        dest = 'train'
    elif idx < train_count + val_count:
        dest = 'valid'
    else:
        dest = 'test'

    # 画像ファイルのコピー
    shutil.copy(os.path.join(dataset_dir, image_file), f'/Users/bruce.young/Downloads/colony_v23/{dest}/images/{image_file}')

    # 対応するラベルファイルもコピー
    label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), f'/Users/bruce.young/Downloads/colony_v23/{dest}/labels/{label_file}')

print(f'Train: {train_count}枚, Validation: {val_count}枚, Test: {test_count}枚 に分割しました。')
