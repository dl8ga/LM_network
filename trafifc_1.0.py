import pandas as pd
import numpy as np
import matplotlib
import torch
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, precision_score
from sklearn.utils import resample

BATCH_SIZE = 64
torch.manual_seed(42)
np.random.seed(42)

class Teacher(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Teacher,self).__init__()

        self.model = nn.Sequential(

             nn.Linear(input_dim,16),
             nn.BatchNorm1d(16),
             nn.ReLU(),
             nn.Dropout(0.5),

             nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def data_processing(data):
    data = data.copy()
    for col in data.columns:
        if data[col].dtype!=object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(data[col].dtype)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max<np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
                elif c_min > np.iinfo(np.int128).min and c_max<np.iinfo(np.int128).max:
                    data[col] = data[col].astype(np.int128)
            else:
                if c_min > np.finfo(np.float32).min and c_max<np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max<np.finfo(np.float64).max:
                    data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype('category')
    return data

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(df, test_size=0.4,random_state=rstate, shuffle=shuffle,stratify=strat)

    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=rstate, shuffle=shuffle,stratify=strat)

    return (train_set, val_set, test_set)

def remove_label(df, label_name):
    X=df.drop(label_name, axis=1)
    y=df[label_name].copy()
    return (X,y)

def optimiz(file_paths, sample=1000):
    target_counts = {
        "Benign": sample,
        "Malicious": sample,
        "Malicious   Attack": sample,
        "Malicious   C&C": sample,
        "Malicious   DDoS": sample,
        "Malicious   PartOfAHorizontalPortScan": sample,
        "Malicious   FileDownload": sample
    }
    
    result_data = pd.DataFrame()
    
    for f in file_paths:
        time_data = pd.read_csv(f, delimiter="|", low_memory=False)
        # print("Уникальные метки в файле:", time_data["label"].unique())

        for label in list(target_counts.keys()):
            if target_counts[label] <= 0:
                continue
                
            label_data = time_data[time_data["label"] == label]
            available_samples = len(label_data)
            needed_samples = target_counts[label]
            
            if available_samples > 0:
                # Берем минимальное из: нужного количества или доступного
                take_samples = min(needed_samples, available_samples)
                sampled_data = label_data.sample(take_samples, random_state=42)
                result_data = pd.concat([result_data, sampled_data], ignore_index=True)
                target_counts[label] -= take_samples
                
                
                # Если собрали достаточно образцов для всех меток, выходим
                if all(count <= 0 for count in target_counts.values()):
                    return result_data
    
    # Фильтруем метки, для которых собрали достаточно образцов
    valid_labels = [label for label, count in target_counts.items() if count == 0]
    
    if not valid_labels:
        return pd.DataFrame()
    
    return result_data[result_data['label'].isin(valid_labels)].copy()

#все в одну бд
file_paths = ["CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-3-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-8-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-9-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-20-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-21-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-34-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-35-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-42-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-44-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-48-1conn.log.labeled.csv",
            "CTU-IoT-Malware-Capture-60-1conn.log.labeled.csv"]

# all_data = pd.concat([pd.read_csv(f, delimiter="|") for f in file_paths], ignore_index=True)
    
all_data = optimiz(file_paths,2000)
all_data.replace('-', pd.NA, inplace=True)

data = all_data.dropna(axis=1,how="all").copy()

columns_to_drop = ['service', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents', 'duration', 'ts','uid', 'missed_bytes']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

data = data_processing(data)


data.drop(['id.orig_h', 'id.resp_h',"detailed-label"],axis=1, inplace=True )

print("Распределение меток после фильтрации:")
print(data['label'].value_counts())


categorical_columns_to_label_encoder = ['proto', 'conn_state', 'history']
for column in categorical_columns_to_label_encoder:
    categorical_label_encoder = LabelEncoder()
    data[column] = categorical_label_encoder.fit_transform(data[column])


numeric_columns = ['id.orig_p', 'id.resp_p', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column])

train_set, val_set, test_set = train_val_test_split(data, stratify='label')
X_train, y_train = remove_label(train_set, 'label')
X_val, y_val = remove_label(val_set, 'label')
X_test, y_test = remove_label(test_set, 'label')

encoder = LabelEncoder()
y_train_encoder = encoder.fit_transform(y_train)
y_val_encoder = encoder.transform(y_val)
y_test_encoder = encoder.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

input_dim = X_test_scaled.shape[1]
num_classes = len(encoder.classes_)
teacher = Teacher(input_dim, num_classes)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoder, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_encoder, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoder, dtype=torch.long)


assert len(X_train_tensor) == len(y_train_tensor)
assert len(X_val_tensor) == len(y_val_tensor)
assert len(X_test_tensor) == len(y_test_tensor)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimazer = opt.Adam(teacher.parameters(), lr=0.0005, weight_decay=0.0005)
loss_funct = nn.CrossEntropyLoss()

class EarlyStopping:
    def __init__(self, patient=5, min_delta=0.001):
        self.patient=patient
        self.min_delta= min_delta
        self.best_loss = float("inf")
        self.counter = 0
    
    def __call__(self,val_loss):
        if val_loss<self.best_loss - self.min_delta:
            self.best_loss=val_loss
            self.counter = 0
        else:
            self.counter+=1
        return self.counter>=self.patient
    
class ReduceLR:
    def __init__(self, optimazer,factor=0.5,patient=3,min_lr=1e-6):
        self.optimazer=optimazer
        self.factor=factor
        self.patient=patient
        self.min_lr=min_lr
        self.best_loss=float("inf")
        self.counter=0
    def __call__(self, val_loss):
        if val_loss<self.best_loss:
            self.best_loss=val_loss
            self.counter = 0
        else:
            self.counter+=1
        if self.counter>=self.patient:
            for param_group in self.optimazer.param_groups:
                new_lr=max(param_group["lr"]*self.factor,self.min_lr)
                param_group['lr'] = new_lr
            self.counter=0

early_stopping = EarlyStopping(patient=5)
reduceLR = ReduceLR(optimazer,factor=0.5,patient=3)



print("start learning... /")
for epoch in range(100):
    teacher.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:

        optimazer.zero_grad()

        output = teacher(batch_x)
        loss = loss_funct(output, batch_y)

        loss.backward()
        optimazer.step()

        train_loss +=loss.item()*batch_x.size(0)

    train_loss /= len(train_loader.dataset) 

    teacher.eval()
    val_loss =0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            val_output = teacher(batch_x)

            val_loss += loss_funct(val_output,batch_y).item()*batch_x.size(0)

    val_loss/=len(val_loader.dataset)
    
    if early_stopping(val_loss):
    
        print(f"Раняя остановка на {epoch}-й эпохе")
        break

    reduceLR(val_loss)

    print(f"Эпоха {epoch} | Потери: {loss:.4f} | Вал. потери: {val_loss:.4f}")

teacher.eval()
with torch.no_grad():
    y_pred = teacher(X_test_tensor).argmax(dim=1)
    print(f"Accuracy: {accuracy_score(y_test_tensor, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test_tensor, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_test_tensor, y_pred,average='macro'):.4f}")
    print("\nF1-scores:")
    print(f"Micro: {f1_score(y_test_tensor, y_pred, average='micro'):.4f}")
    print(f"Macro: {f1_score(y_test_tensor, y_pred, average='macro'):.4f}")
    print(f"Weighted: {f1_score(y_test_tensor, y_pred, average='weighted'):.4f}")
    

# # Вывести всю архитектуру модели с параметрами
# print(teacher)

# # Посмотреть веса и смещения (biases) для всех слоёв
# for name, param in teacher.named_parameters():
#     print(f"Слой: {name} | Размер: {param.size()} | Значения:\n{param.data}\n")