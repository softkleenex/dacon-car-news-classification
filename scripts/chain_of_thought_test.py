#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain of Thought (CoT) 프롬프팅 테스트
추론 과정을 유도하는 프롬프트와 일반 프롬프트 비교
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
from openai import OpenAI

try:
    from config import API_KEY, MODEL, TEMPERATURE, API_DELAY
except ImportError:
    print("ERROR: config.py 파일을 찾을 수 없습니다.")
    sys.exit(1)

# CoT 프롬프트 vs 일반 프롬프트
COT_PROMPTS = [
    # Baseline (non-CoT)
    {
        "name": "BASELINE",
        "prompt": "자동차 아니면 0, 맞으면 1.",
        "type": "standard"
    },
    
    # Chain of Thought - 영어
    {
        "name": "COT_ENGLISH",
        "prompt": "Let's think step by step. If the news is about cars, output 1. Otherwise, output 0.",
        "type": "cot"
    },
    
    # Chain of Thought - 한국어
    {
        "name": "COT_KOREAN",
        "prompt": "단계별로 생각해봅시다. 자동차 관련 뉴스면 1, 아니면 0을 출력하세요.",
        "type": "cot"
    },
    
    # 분석 후 답변
    {
        "name": "ANALYZE_FIRST",
        "prompt": "주제를 분석한 후 답하세요. 자동차면 1, 아니면 0",
        "type": "cot"
    },
    
    # 신중한 판단
    {
        "name": "THINK_CAREFULLY",
        "prompt": "Think carefully before answering. Car-related: 1, Not car-related: 0",
        "type": "cot"
    },
    
    # 2단계 프로세스
    {
        "name": "TWO_STEP",
        "prompt": "Step 1: Identify the topic. Step 2: Output 1 if car-related, 0 if not.",
        "type": "process"
    },
    
    # 명확한 지시 + CoT
    {
        "name": "CLEAR_COT",
        "prompt": "먼저 주제를 파악하고, 자동차 관련이면 1, 아니면 0만 출력",
        "type": "cot"
    }
]

def test_prompt(prompt, df, client, sample_size=None):
    """프롬프트 테스트"""
    if sample_size:
        test_df = df.head(sample_size)
    else:
        test_df = df
    
    predictions = []
    outputs = []
    
    for i, row in test_df.iterrows():
        msg = f"제목: {row['title']}\n내용: {str(row['content'])[:300]}"
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": msg}
                ],
                temperature=TEMPERATURE,
                max_tokens=50  # CoT는 더 긴 출력 가능
            )
            
            result = response.choices[0].message.content.strip()
            
            # 파싱 (CoT는 설명 포함 가능)
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
            if i < 3:
                outputs.append(result[:50])
            
            time.sleep(API_DELAY)
            
        except Exception as e:
            predictions.append(0)
    
    correct = sum(p == l for p, l in zip(predictions, test_df['label']))
    accuracy = correct / len(test_df)
    
    # 점수 계산
    length = len(prompt.encode('utf-8'))
    length_score = np.sqrt(1 - (length / 3000) ** 2)
    final_score = 0.9 * accuracy + 0.1 * length_score
    
    return accuracy, final_score, outputs

def main():
    print("="*80)
    print("Chain of Thought (CoT) 프롬프팅 테스트")
    print("="*80)
    
    # API 클라이언트 초기화
    client = OpenAI(api_key=API_KEY)
    
    # 데이터 로드
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    print(f"\n샘플: {len(df)}개 (자동차: {sum(df['label']==1)}, 비자동차: {sum(df['label']==0)})")
    
    # 빠른 테스트 또는 전체 테스트
    test_size = 20  # 빠른 테스트용
    print(f"\n테스트 크기: {test_size}개 샘플 (빠른 테스트)")
    
    results = []
    print("\n테스트 시작...")
    print("-"*80)
    
    for item in COT_PROMPTS:
        name = item['name']
        prompt = item['prompt']
        ptype = item['type']
        
        print(f"\n[{ptype.upper()}] {name}")
        print(f"프롬프트: '{prompt[:50]}...'")
        
        accuracy, score, outputs = test_prompt(prompt, df, client, test_size)
        
        results.append({
            'name': name,
            'type': ptype,
            'prompt': prompt,
            'accuracy': accuracy,
            'score': score,
            'outputs': outputs
        })
        
        print(f"정확도: {accuracy:.1%} | 점수: {score:.4f}")
        if outputs:
            print(f"샘플 출력: {outputs[0][:30]}...")
    
    # 결과 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 요약
    print("\n" + "="*80)
    print("결과 요약 (점수 순)")
    print("="*80)
    
    print(f"\n{'순위':<4} {'이름':<20} {'타입':<10} {'정확도':>8} {'점수':>8}")
    print("-"*50)
    
    for i, r in enumerate(results[:5], 1):
        print(f"{i:<4} {r['name']:<20} {r['type']:<10} {r['accuracy']:>7.1%} {r['score']:>8.4f}")
    
    # 타입별 평균
    print("\n" + "="*80)
    print("타입별 비교")
    print("="*80)
    
    type_stats = {}
    for r in results:
        if r['type'] not in type_stats:
            type_stats[r['type']] = []
        type_stats[r['type']].append(r['accuracy'])
    
    print("\n타입별 평균 정확도:")
    for t, accs in type_stats.items():
        avg = sum(accs) / len(accs)
        print(f"  {t:<12}: {avg:.1%} ({len(accs)}개 프롬프트)")
    
    # 결론
    print("\n" + "="*80)
    print("핵심 발견")
    print("="*80)
    
    baseline = next((r for r in results if r['name'] == 'BASELINE'), None)
    best_cot = max((r for r in results if r['type'] == 'cot'), key=lambda x: x['score'], default=None)
    
    if baseline and best_cot:
        print(f"""
1. Baseline vs Best CoT:
   - Baseline: {baseline['accuracy']:.1%} (점수: {baseline['score']:.4f})
   - Best CoT: {best_cot['accuracy']:.1%} (점수: {best_cot['score']:.4f})
   - 차이: {(best_cot['accuracy'] - baseline['accuracy'])*100:+.1f}%

2. 결론:
   - CoT는 복잡한 추론에 유용하지만 단순 이진 분류에는 과도할 수 있음
   - 길이 페널티로 인해 점수가 낮아질 수 있음
   - 간단명료한 프롬프트가 더 효과적일 수 있음
""")

if __name__ == "__main__":
    main()