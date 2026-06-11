#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 vs 영어 프롬프트 비교 테스트
동일한 의미의 프롬프트를 두 언어로 테스트하여 성능 차이 분석
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
from openai import OpenAI

from scripts.settings import (
    API_DELAY,
    FULL_TEST,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    TEST_SAMPLE_SIZE,
    require_api_key,
)

# 언어별 프롬프트 쌍
PROMPT_PAIRS = [
    {
        "name": "Simple Condition",
        "korean": "자동차면 1, 아니면 0",
        "english": "Car: 1, Others: 0"
    },
    {
        "name": "If-Else",
        "korean": "자동차 관련이면 1, 아니면 0",
        "english": "If car-related 1, else 0"
    },
    {
        "name": "News Classification",
        "korean": "자동차 뉴스면 1, 다른 뉴스면 0",
        "english": "Car news: 1, Other news: 0"
    },
    {
        "name": "Not-Then",
        "korean": "자동차 아니면 0, 맞으면 1",
        "english": "Not car: 0, Car: 1"
    },
    {
        "name": "Vehicle Variant",
        "korean": "차량이면 1, 아니면 0",
        "english": "Vehicle: 1, Others: 0"
    }
]

def test_prompt(prompt, df, client):
    """프롬프트 테스트"""
    predictions = []
    
    for i, row in df.iterrows():
        msg = f"제목: {row['title']}\n내용: {str(row['content'])[:300]}"
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": msg}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            result = response.choices[0].message.content.strip()
            
            # 결과 파싱
            if '1' in result and '0' in result:
                last_1 = result.rfind('1')
                last_0 = result.rfind('0')
                pred = 1 if last_1 > last_0 else 0
            elif '1' in result:
                pred = 1
            elif '0' in result:
                pred = 0
            else:
                pred = 0
            
            predictions.append(pred)
            time.sleep(API_DELAY)
            
        except Exception as e:
            predictions.append(0)
    
    correct = sum(p == l for p, l in zip(predictions, df['label']))
    accuracy = correct / len(df)
    
    # 카테고리별 정확도
    car_correct = sum(p == 1 and l == 1 for p, l in zip(predictions, df['label']))
    non_car_correct = sum(p == 0 and l == 0 for p, l in zip(predictions, df['label']))
    
    car_total = sum(df['label'] == 1)
    non_car_total = sum(df['label'] == 0)
    
    car_acc = car_correct / car_total if car_total > 0 else 0
    non_car_acc = non_car_correct / non_car_total if non_car_total > 0 else 0
    
    return accuracy, car_acc, non_car_acc

def main():
    print("="*80)
    print("한국어 vs 영어 프롬프트 비교 테스트")
    print("="*80)
    
    # API 클라이언트 초기화
    client = OpenAI(api_key=require_api_key())
    
    # 데이터 로드
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 테스트 크기 결정
    if FULL_TEST:
        test_df = df
        print(f"\n전체 테스트: {len(test_df)}개 샘플")
    else:
        test_df = df.head(TEST_SAMPLE_SIZE)
        print(f"\n빠른 테스트: {len(test_df)}개 샘플")
    
    print(f"자동차: {sum(test_df['label']==1)}, 비자동차: {sum(test_df['label']==0)}")
    
    # 테스트 실행
    results = []
    print("\n테스트 시작...")
    print("-"*80)
    
    for pair in PROMPT_PAIRS:
        print(f"\n테스트: {pair['name']}")
        print(f"  한국어: '{pair['korean']}'")
        print(f"  영어: '{pair['english']}'")
        
        # 한국어 테스트
        print("  한국어 테스트 중...", end="")
        k_acc, k_car, k_non = test_prompt(pair['korean'], test_df, client)
        print(f" {k_acc:.1%}")
        
        # 영어 테스트
        print("  영어 테스트 중...", end="")
        e_acc, e_car, e_non = test_prompt(pair['english'], test_df, client)
        print(f" {e_acc:.1%}")
        
        results.append({
            'name': pair['name'],
            'korean': pair['korean'],
            'english': pair['english'],
            'korean_acc': k_acc,
            'english_acc': e_acc,
            'difference': k_acc - e_acc,
            'korean_car': k_car,
            'english_car': e_car,
            'korean_non': k_non,
            'english_non': e_non
        })
        
        print(f"  차이: 한국어가 {(k_acc - e_acc)*100:+.1f}% {'높음' if k_acc > e_acc else '낮음'}")
    
    # 결과 요약
    print("\n" + "="*80)
    print("결과 요약")
    print("="*80)
    
    print(f"\n{'테스트':<20} {'한국어':>10} {'영어':>10} {'차이':>10}")
    print("-"*50)
    
    for r in results:
        print(f"{r['name']:<20} {r['korean_acc']:>9.1%} {r['english_acc']:>9.1%} {r['difference']*100:>+9.1f}%")
    
    # 평균 계산
    avg_korean = sum(r['korean_acc'] for r in results) / len(results)
    avg_english = sum(r['english_acc'] for r in results) / len(results)
    avg_diff = avg_korean - avg_english
    
    print("-"*50)
    print(f"{'평균':<20} {avg_korean:>9.1%} {avg_english:>9.1%} {avg_diff*100:>+9.1f}%")
    
    # 카테고리별 분석
    print("\n" + "="*80)
    print("카테고리별 분석")
    print("="*80)
    
    print("\n자동차 뉴스 정확도:")
    for r in results:
        print(f"  {r['name']}: 한국어 {r['korean_car']:.1%} vs 영어 {r['english_car']:.1%}")
    
    print("\n비자동차 뉴스 정확도:")
    for r in results:
        print(f"  {r['name']}: 한국어 {r['korean_non']:.1%} vs 영어 {r['english_non']:.1%}")
    
    # 결론
    print("\n" + "="*80)
    print("결론")
    print("="*80)
    
    if avg_diff > 0:
        print(f"""
한국어 프롬프트가 영어보다 평균 {avg_diff*100:.1f}% 높은 성능을 보였습니다.

이유:
1. 한국 뉴스 데이터의 언어적 특성
2. GPT 모델의 한국어 문맥 이해
3. 번역 과정에서의 의미 손실 최소화

추천: 한국 데이터에는 한국어 프롬프트 사용
""")
    else:
        print(f"""
영어 프롬프트가 한국어보다 평균 {abs(avg_diff)*100:.1f}% 높은 성능을 보였습니다.

가능한 이유:
1. GPT 모델의 영어 학습 데이터가 더 풍부
2. 영어의 간결한 표현
3. 특정 도메인에서 영어가 더 명확

하지만 일반적으로 한국 데이터에는 한국어가 유리합니다.
""")

if __name__ == "__main__":
    main()
