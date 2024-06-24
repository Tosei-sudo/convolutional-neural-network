# coding: utf-8

import numpy as np
import itertools

def initial_segmentation(image, scale=100, sigma=0.8, min_size=50):
    # 簡易的なセグメンテーション: 各ピクセルを独立したセグメントとする
    height, width = image.shape[1], image.shape[2]
    segments = np.arange(height * width).reshape(height, width)
    return segments

def calculate_similarity(region1, region2, image):
    # 類似度計算の簡易実装: 色の類似度を基に計算
    color1 = np.mean(image[:, region1[:, 0], region1[:, 1]], axis=1)
    color2 = np.mean(image[:, region2[:, 0], region2[:, 1]], axis=1)
    return np.linalg.norm(color1 - color2)

def selective_search(image):
    height, width = image.shape[1], image.shape[2]
    
    # 初期セグメンテーション
    segments = initial_segmentation(image)
    
    # 隣接セグメントペアの初期化
    regions = {}
    for y in range(height):
        for x in range(width):
            label = segments[y, x]
            if label not in regions:
                regions[label] = []
            regions[label].append((y, x))
    print("end initial_segmentation")

    # セグメントペアの類似度計算
    pairs = list(itertools.combinations(regions.keys(), 2))
    similarities = {}
    for (region1, region2) in pairs:
        region1_coords = np.array(regions[region1])
        region2_coords = np.array(regions[region2])
        similarities[(region1, region2)] = calculate_similarity(region1_coords, region2_coords, image)
    print("end similarity")

    # 類似度に基づいてセグメントをマージ
    threshold = 30
    while similarities:
        most_similar_pair = max(similarities, key=similarities.get)
        if similarities[most_similar_pair] < threshold:
            break
        region1, region2 = most_similar_pair
        
        # マージ
        regions[region1].extend(regions[region2])
        del regions[region2]
        
        # 類似度の更新
        similarities = {(r1, r2): calculate_similarity(np.array(regions[r1]), np.array(regions[r2]), image)
                        for r1, r2 in itertools.combinations(regions.keys(), 2)}
    print("end merge")

    # 候補領域の抽出
    windows = []
    for region in regions.values():
        y_coords, x_coords = zip(*region)
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        windows.append((x1, y1, x2, y2))

    return windows
