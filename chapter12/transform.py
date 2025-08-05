from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 학습 데이터에 적용할 변환(augmentation) 정의
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # 이미지 크기 조정
    transforms.RandomRotation(30),      # 랜덤 회전 (30도 이내)
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 색상 jittering
    transforms.ToTensor(),              # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화
])

# 검증/테스트 데이터에 적용할 변환 정의 (일반적으로 증식을 포함하지 않음)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform # transform 객체 저장
        self.image_files = ...
        
    def __getitem__(self, idx):
        # 1. 인덱스에 해당하는 데이터(이미지, 라벨) 로드
        image = Image.open(self.image_files[idx]).convert('RGB')
        label = self.labels[idx]

        # 2. transform이 존재하면, 로드된 이미지에 transform 적용
        if self.transform:
            image = self.transform(image) # <-- 이 부분에서 변환이 실행됨

        return image, label
    
train_dataset = CustomDataset(data_path='path/to/train_data', transform=train_transforms)
val_dataset = CustomDataset(data_path='path/to/val_data', transform=val_transforms)