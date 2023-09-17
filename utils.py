import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# 이미지를 읽는 함수를 정의
def get_image(p:str):
    return Image.open(p).convert("RGB")

class baseDataset(Dataset):
    name2idx = {
        'airplane' : 0, 'automobile' : 1, 'bird' : 2,
        'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6,
        'horse' : 7, 'ship' : 8, 'truck' : 9
    }
    
    def __init__(self, root:str="./datasets/CIFAR-10", istrain:bool=True):
        super().__init__()
        root = os.path.join(root, 'train') if istrain else os.path.join(root, 'val')
        
        # 데이터 리스트 생성
        data_list = []
        for class_name in os.listdir(root):  # 수정된 부분
            class_dir = os.path.join(root, class_name)  # 수정된 부분
            for img in os.listdir(class_dir):  # 수정된 부분
                img_path = os.path.join(class_dir, img)
                data_list.append((self.name2idx[class_name], img_path))
        self.data_list = data_list
        
        # 훈련과 검증과정에서 서로 다른 augmentatation 기법을 사용
        if istrain:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),  # 랜덤으로 이미지를 수평으로 뒤집음
                T.RandomCrop(32, padding=4),  # 이미지를 랜덤하게 자르고 크기를 32x32로 조정
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 이미지의 색감을 랜덤으로 조정
                T.RandomRotation(15),  # 이미지를 랜덤하게 회전
                T.ToTensor(),  # 이미지를 텐서로 변환
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(32),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx:int):
        # 인덱스에 해당하는 데이터 반환
        number, img_path = self.data_list[idx]
        img_obj = get_image(img_path)
        img_tensor = self.transform(img_obj)
        
        return img_tensor, number
    
if __name__ == "__main__":
    # test baseDataset Class
    train_dataset, val_dataset = baseDataset("./datasets/CIFAR-10", True), baseDataset("./datasets/CIFAR-10", False)
    print(f"# Length of data, TRAIN : {len(train_dataset)}, VAL : {len(val_dataset)}")
    
    # test tensor shape
    tensor, number = train_dataset[200]
    print(f"# Tensor Shape(unbatched) : {tensor.shape}")