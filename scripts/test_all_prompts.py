#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DACON 자동차 뉴스 분류 - 프롬프트 전체 테스트
다양한 프롬프트를 체계적으로 테스트하고 성능을 비교합니다.

Author: softkleenex
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
import json
from openai import OpenAI

from scripts.settings import API_DELAY, MAX_TOKENS, MODEL, TEMPERATURE, require_api_key

# 테스트할 프롬프트 목록
TEST_PROMPTS = [
    # 최고 성능 프롬프트들
    "자동차 아니면 0 맞으면 1",
    "주제를 분석한 후 답하세요. 자동차면 1, 아니면 0",
    "먼저 주제를 파악하고, 자동차 관련이면 1, 아니면 0만 출력",
    "Step 1: Identify the topic. Step 2: Output 1 if car-related, 0 if not.",
    
    # 기존 제출 프롬프트
    "자동차 아니면 0, 맞으면 1.",
    "차량 아니면 0, 맞으면 1.",
    
    # 간단한 프롬프트
    "자동차면 1, 아니면 0",
    "자동차 1, 기타 0",
    
    # 영어 프롬프트
    "Car news: 1, Others: 0",
    "If car-related output 1, else 0",
    
    # 실패 케이스 (참고용)
    "자동차 1",
    "1 자동차"
]

def test_prompt(prompt, df, client):
    """
    특정 프롬프트로 전체 샘플 테스트
    """
    predictions = []
    outputs = []
    
    for idx, row in df.iterrows():
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
            if idx < 3:  # 처음 3개 샘플 출력 저장
                outputs.append(result[:30])
            
            time.sleep(API_DELAY)
            
        except Exception as e:
            print(f"  ERROR at sample {idx}: {str(e)}")
            predictions.append(0)
    
    # 점수 계산
    correct = sum(p == l for p, l in zip(predictions, df['label']))
    accuracy = correct / len(df)
    
    # 길이 점수 계산
    prompt_bytes = len(prompt.encode('utf-8'))
    length_score = np.sqrt(1 - (prompt_bytes / 3000) ** 2)
    final_score = 0.9 * accuracy + 0.1 * length_score
    
    # 카테고리별 정확도
    car_indices = df[df['label']==1].index
    non_car_indices = df[df['label']==0].index
    
    car_correct = sum(predictions[idx] == 1 for idx in range(len(df)) if df.iloc[idx]['label'] == 1)
    non_car_correct = sum(predictions[idx] == 0 for idx in range(len(df)) if df.iloc[idx]['label'] == 0)
    
    car_acc = car_correct / len(car_indices) if len(car_indices) > 0 else 0
    non_car_acc = non_car_correct / len(non_car_indices) if len(non_car_indices) > 0 else 0
    
    return {
        'prompt': prompt,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(df),
        'car_accuracy': car_acc,
        'non_car_accuracy': non_car_acc,
        'score': final_score,
        'prompt_bytes': prompt_bytes,
        'sample_outputs': outputs,
        'predictions': predictions
    }

def main():
    print("="*80)
    print("DACON 자동차 뉴스 분류 - 전체 프롬프트 테스트")
    print("="*80)
    
    # API 클라이언트 초기화
    client = OpenAI(api_key=require_api_key())
    
    # 데이터 로드
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples.csv')
    if not os.path.exists(data_path):
        print(f"ERROR: 데이터 파일을 찾을 수 없습니다: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"\n데이터 로드 완료: {len(df)}개 샘플")
    print(f"  자동차 관련: {sum(df['label']==1)}개")
    print(f"  비자동차: {sum(df['label']==0)}개")
    
    # 테스트 실행
    print("\n프롬프트 테스트 시작...")
    print("-"*80)
    
    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] 테스트 중: '{prompt[:40]}...'")
        
        result = test_prompt(prompt, df, client)
        results.append(result)
        
        print(f"  정확도: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        print(f"  점수: {result['score']:.4f}")
        print(f"  자동차: {result['car_accuracy']:.1%}, 비자동차: {result['non_car_accuracy']:.1%}")
    
    # 결과 정렬 및 출력
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*80)
    print("테스트 결과 요약 (점수 순)")
    print("="*80)
    print(f"\n{'순위':<4} {'프롬프트':<40} {'정확도':>8} {'점수':>8} {'길이':>6}")
    print("-"*70)
    
    for i, r in enumerate(results[:10], 1):
        prompt_display = r['prompt'][:35] + '...' if len(r['prompt']) > 35 else r['prompt']
        print(f"{i:<4} {prompt_display:<40} {r['accuracy']:>7.1%} {r['score']:>8.4f} {r['prompt_bytes']:>6}")
    
    # 결과 저장
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'test_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 {output_path}에 저장되었습니다.")
    
    # 최고 프롬프트 추천
    best = results[0]
    print("\n" + "="*80)
    print("추천 프롬프트")
    print("="*80)
    print(f"\n최고 성능: '{best['prompt']}'")
    print(f"  예상 점수: {best['score']:.4f}")
    print(f"  정확도: {best['accuracy']:.1%}")
    print(f"  길이: {best['prompt_bytes']} bytes")

if __name__ == "__main__":
    main()
