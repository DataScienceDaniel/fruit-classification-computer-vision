
    
    # Original file is located at
        # https://colab.research.google.com/drive/1JV254KhEbcmoKXaLtrxvacwKy2Ll5Wc1
    
    ### Importações Necessárias
    """
    
    pip install ultralytics
    
    import kagglehub
    import pandas as pd
    import os, glob, json, csv, itertools, math, random
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    from sklearn.cluster import KMeans
    from collections import Counter
    from matplotlib import colors
    import shutil
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
    from pathlib import Path
    from PIL import Image
    import time
    
    from ultralytics import YOLO
    
    """### Baixar o dataset do Kaggle e criar nossos DataFrames de Treino e Teste"""
    
    path = kagglehub.dataset_download("kvnpatel/fruits-vegetable-detection-for-yolov4")
    
    os.listdir(path)
    
    """### Função para construir o dataframe de imagens"""
    
    def build_image_df(base_dir):
        data = []
    
        for class_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        fpath = os.path.join(class_path, fname)
                        data.append({"class": class_name,
                                     "filename": fname,
                                     "path": fpath})
        return pd.DataFrame(data)
    
    dir_1 = os.path.join(path, "obj (1)")
    df_1 = build_image_df(dir_1)
    
    dir_2 = os.path.join(path, "test")
    df_2  = build_image_df(dir_2)
    
    df = pd.concat([df_1, df_2])
    
    df
    
    del df['class']
    
    """### Criação da coluna de sacola para separar imagens com frutas em sacolas"""
    
    df['Bag'] = np.select([df['filename'].str.contains('wb', case=False, na=False), df['filename'].str.contains('wob', case=False, na=False)], ['Yes','No'], default=pd.NA)
    
    df['Bag'].value_counts()
    
    """### Extração de rótulos únicos"""
    
    def extract_label(name):
        # 1️⃣ Padrões com _)
        match = re.search(r'^\d+_\d+_([a-zA-Z]+)_', name)
        if match:
            return match.group(1).lower()
    
        # 2️⃣ Padrões com -)
        match = re.search(r'^([a-zA-Z]+(?:\s[a-zA-Z]+)?)\s*-\s*\d+', name)
        if match:
            return match.group(1).strip().lower()
    
        # 3️⃣ Se não for nenhum desses padrões, encontre a primeira palavra
        match = re.search(r'([a-zA-Z]+)', name)
        if match:
            return match.group(1).lower()
    
        return None
    
    df['Label'] = df['filename'].apply(extract_label)
    
    df['Label'].value_counts()
    
    """### Extração de rótulos com e sem sacola"""
    
    def build_final_label(row):
        base = row["Label"].lower().replace(" ", "_")
        bag = row["Bag"]
    
        # Se não tem info de sacola, volta só o rótulo base
        if pd.isna(bag):
            return base
    
        if bag == "Yes":
            return f"{base}_with_bag"
        elif bag == "No":
            return f"{base}_without_bag"
        else:
            return base
    
    df["FinalLabel"] = df.apply(build_final_label, axis=1)
    df["FinalLabel"].value_counts()
    
    df
    
    """### Análises iniciais, visualizando imagens"""
    
    image_path_tomato_wb = df.iloc[1]['path']
    
    image_tomato_wb = cv2.imread(image_path_tomato_wb)
    plt.imshow(image_tomato_wb)
    plt.title("Tomato with Bag")
    plt.axis('off')
    plt.show()
    
    image_path_tomato_wob = df.iloc[646]['path']
    image_tomato_wob = cv2.imread(image_path_tomato_wob)
    plt.imshow(image_tomato_wob)
    plt.title("Loaded Image")
    plt.axis('off')
    plt.show()
    
    image_rgb_tomato_wob = cv2.cvtColor(image_tomato_wob, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb_tomato_wob)
    plt.title("Loaded Image")
    plt.axis('off')
    plt.show()
    
    del df['filename']
    del df['Bag']
    del df['Label']
    
    """### Separação Treino e Teste"""
    
    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['FinalLabel'])
    
    def copy_images(sub_df, subset_name):
        for _, row in sub_df.iterrows():
            label_folder = f"dataset/{subset_name}/{row['FinalLabel']}"
            os.makedirs(label_folder, exist_ok=True)
            shutil.copy(row['path'], os.path.join(label_folder, os.path.basename(row['path'])))
    
    copy_images(train_df, "train")
    copy_images(val_df, "val")
    
    train_df = train_df.drop(columns=['FinalLabel'])
    val_df = val_df.drop(columns=['FinalLabel'])
    
    """### Modelo Yolov8s-cls"""
    
    model = YOLO("yolov8s-cls.pt")
    model.train(data="dataset", epochs=50, imgsz=224, batch=16)
    
    """### Imagens de validação"""
    
    def latest_run_dir():
        expdirs = sorted(glob.glob("runs/classify/*"), key=os.path.getmtime)
        return expdirs[-1]
    
    run_dir = latest_run_dir()
    print("Using run:", run_dir)
    
    
    best_ckpt = os.path.join(run_dir, "weights", "best.pt")
    model = YOLO(best_ckpt)
    
    
    names = model.names if hasattr(model, "names") else model.model.names
    idx2name = {i: n for i, n in names.items()} if isinstance(names, dict) else {i:n for i,n in enumerate(names)}
    
    
    val_root = Path("dataset/val")
    val_images = []
    for c in os.listdir(val_root):
        cdir = val_root / c
        if cdir.is_dir():
            for img in cdir.rglob("*.*"):
                if img.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                    val_images.append((str(img), c))
    print("Val images:", len(val_images))
    
    
    y_true, y_pred, y_prob = [], [], []
    pred_rows = []
    
    """### Teste de Robustez visual (translucência / iluminação)"""
    
    def stress_test_image(path, show=True):
        img = cv2.imread(path)
        if img is None:
            print("Could not read image:", path)
            return
    
        tests = {
            "original": img,
            "bright": cv2.convertScaleAbs(img, alpha=1.3, beta=20),
            "dark": cv2.convertScaleAbs(img, alpha=0.7, beta=-20),
            "occlusion": img.copy()
        }
    
        # fake occlusion block
        h, w, _ = img.shape
        tests["occlusion"][h//4: h//2, w//4:w//2] = 0
    
        preds = {}
    
        # run inference on each transformed image
        for name, im in tests.items():
            r = model.predict(im, imgsz=160, verbose=False)[0]
            cls = r.names[r.probs.top1]
            conf = float(r.probs.top1conf)
            preds[name] = (cls, conf)
            print(f"{name:10s} → {cls} ({conf:.2f})")
    
        # visualize in a grid
        if show:
            n = len(tests)
            cols = 3
            rows = math.ceil(n / cols)
            plt.figure(figsize=(4*cols, 4*rows))
    
            for i, (name, im) in enumerate(tests.items(), start=1):
                img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                plt.subplot(rows, cols, i)
                plt.imshow(img_rgb)
                cls, conf = preds[name]
                plt.title(f"{name} → {cls} ({conf:.2f})")
                plt.axis("off")
    
            plt.tight_layout()
            plt.show()
    
    # call it
    stress_test_image(val_images[0][0])
    
    """### Teste de Medir tempo de inferência (ms)"""
    
    def benchmark_latency(model, sample_path, runs=30):
        times = []
        for _ in range(runs):
            t0 = time.time()
            _ = model.predict(sample_path, imgsz=160, verbose=False)
            times.append((time.time()-t0)*1000)  # ms
        print(f"Latency (mean over {runs} runs): {np.mean(times):.2f} ms")
        print(f"Best: {np.min(times):.2f} ms | Worst: {np.max(times):.2f} ms")
    
    
    benchmark_latency(model, val_images[0][0])
    
    """### Teste de Medir footprint de memória do modelo"""
    
    model_size_mb = os.path.getsize(best_ckpt) / (1024*1024)
    print(f"Model file size: {model_size_mb:.2f} MB")
    
    """### Acurácia das Classes Geral"""
    
    for path, true_label in val_images:
        r = model.predict(path, imgsz=224, verbose=False)[0]
        probs = r.probs.data.cpu().numpy() if r.probs is not None else None
        pred_idx = int(np.argmax(probs))
        pred_label = idx2name[pred_idx]
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_prob.append(probs)
        pred_rows.append({"path": path, "true": true_label, "pred": pred_label})
    
    
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    
    """### Matriz de Confusão"""
    
    labels_sorted = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    
    
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (YOLO)")
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha="right")
    plt.yticks(range(len(labels_sorted)), labels_sorted)
    
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()
    
    """### Salvar Predições"""
    
    out_csv = os.path.join(run_dir, "val_predictions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","true","pred"])
        w.writeheader()
        w.writerows(pred_rows)
    print("Saved:", out_csv)
    
    """### Gráficos de acurácia e perda"""
    
    results_csv = os.path.join(run_dir, "results.csv")
    hist = pd.read_csv(results_csv)
    
    plt.figure(figsize=(6,4))
    plt.plot(hist["epoch"], hist["metrics/accuracy_top1"], label="val_acc_top1")
    plt.plot(hist["epoch"], hist["metrics/accuracy_top5"], label="val_acc_top5")
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6,4))
    plt.plot(hist["epoch"], hist["train/loss"], label="train_loss")
    if "val/loss" in hist.columns:
        plt.plot(hist["epoch"], hist["val/loss"], label="val_loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    pr, rc, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels_sorted, zero_division=0)
    
    plt.figure(figsize=(7,4))
    x = np.arange(len(labels_sorted))
    w = 0.25
    plt.bar(x - w, pr, width=w, label="Precision")
    plt.bar(x, rc, width=w, label="Recall")
    plt.bar(x + w, f1, width=w, label="F1")
    plt.xticks(x, labels_sorted, rotation=45, ha="right")
    plt.title("Per-class PRF (YOLO)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # pega todas as imagens de validação
    all_images = glob.glob("dataset/val/*/*.jpg") + glob.glob("dataset/val/*/*.png")
    
    # separa com e sem sacola
    bagged_images = [img for img in all_images if re.search(r"wb", img, re.IGNORECASE)]
    nobag_images = [img for img in all_images if re.search(r"wob", img, re.IGNORECASE)]
    
    print(f"Found {len(bagged_images)} bagged images (wb).")
    print(f"Found {len(nobag_images)} non-bag images (wob).")
    
    # garante que temos pelo menos 1 de cada
    if len(bagged_images) == 0 or len(nobag_images) == 0:
        raise ValueError("Need at least one 'wb' and one 'wob' image in dataset/val.")
    
    # escolhe até 2 de cada
    chosen_bag = random.sample(bagged_images, min(2, len(bagged_images)))
    chosen_nobag = random.sample(nobag_images, min(2, len(nobag_images)))
    
    images = chosen_bag + chosen_nobag
    random.shuffle(images)  # embaralha ordem no grid
    
    print("Using images:")
    for p in images:
        print("  ", p)
    
    model = YOLO(os.path.join(run_dir, "weights", "best.pt"))
    
    plt.figure(figsize=(10, 10))
    for i, path in enumerate(images):
        # inferência
        r = model.predict(path, imgsz=160, conf=0.5, verbose=False)[0]
        pred_class = r.names[r.probs.top1]
        conf = float(r.probs.top1conf)
    
        # carrega imagem
        img = cv2.imread(path)
        if img is None:
            print("Could not read image:", path)
            continue
    
        h, w, _ = img.shape
    
        # quadrado consistente e grosso
        box_size = int(min(h, w) / 1.3)
        x1 = int((w - box_size) / 2)
        y1 = int((h - box_size) / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size
        thickness = 50
    
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # detecta se é wb ou wob só pra escrever no título
        if re.search(r"wb", path, re.IGNORECASE):
            bag_status = "with bag"
        elif re.search(r"wob", path, re.IGNORECASE):
            bag_status = "without bag"
        else:
            bag_status = "unknown"
    
        # 2x2 grid
        plt.subplot(2, 2, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"{bag_status} – Pred: {pred_class} ({conf:.2f})")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
