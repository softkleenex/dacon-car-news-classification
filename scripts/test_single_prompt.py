#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 프롬프트 테스트 스크립트
특정 프롬프트를 테스트하고 상세 결과를 제공합니다.

사용법:
python test_single_prompt.py "자동차 아니면 0 맞으면 1"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
from openai import OpenAI

try:
    from config import API_KEY, MODEL, TEMPERATURE, MAX_TOKENS, API_DELAY
except ImportError:
    print("ERROR: config.py 파일을 찾을 수 없습니다.")
    print("config.py.example을 config.py로 복사하고 API 키를 설정해주세요.")
    sys.exit(1)

def test_prompt_detailed(prompt, df, client):
    """
    프롬프트 상세 테스트 및 분석
    """
    predictions = []
    outputs = []
    errors = []
    
    print(f"\n테스트 진행 중...")
    
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
            outputs.append(result)
            
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
            
            # 진행 상황 표시
            if (idx + 1) % 10 == 0:
                print(f"  {idx + 1}/{len(df)} 완료...")
            
            time.sleep(API_DELAY)
            
        except Exception as e:
            print(f"  ERROR at sample {idx}: {str(e)}")
            predictions.append(0)
            outputs.append("ERROR")
            errors.append((idx, str(e)))
    
    return predictions, outputs, errors

def analyze_results(df, predictions, outputs, prompt):
    """
    결과 상세 분석
    """
    # 기본 통계
    correct = sum(p == l for p, l in zip(predictions, df['label']))
    accuracy = correct / len(df)
    
    # 카테고리별 분석
    car_mask = df['label'] == 1
    non_car_mask = df['label'] == 0
    
    car_predictions = [p for i, p in enumerate(predictions) if car_mask.iloc[i]]
    non_car_predictions = [p for i, p in enumerate(predictions) if non_car_mask.iloc[i]]
    
    car_correct = sum(p == 1 for p in car_predictions)
    non_car_correct = sum(p == 0 for p in non_car_predictions)
    
    car_acc = car_correct / len(car_predictions) if car_predictions else 0
    non_car_acc = non_car_correct / len(non_car_predictions) if non_car_predictions else 0
    
    # 점수 계산
    prompt_bytes = len(prompt.encode('utf-8'))
    length_score = np.sqrt(1 - (prompt_bytes / 3000) ** 2)
    final_score = 0.9 * accuracy + 0.1 * length_score
    
    # 오분류 분석
    misclassified = []
    for i, (pred, label) in enumerate(zip(predictions, df['label'])):
        if pred != label:
            misclassified.append({
                'index': i,
                'title': df.iloc[i]['title'],
                'predicted': pred,
                'actual': label,
                'output': outputs[i] if i < len(outputs) else 'N/A'
            })
    
    return {
        'accuracy': accuracy,
        'car_accuracy': car_acc,
        'non_car_accuracy': non_car_acc,
        'score': final_score,
        'prompt_bytes': prompt_bytes,
        'correct': correct,
        'total': len(df),
        'misclassified': misclassified
    }

def main():
    # 커맨드라인 인자 처리
    if len(sys.argv) < 2:
        print("사용법: python test_single_prompt.py \"프롬프트\"")
        print("예시: python test_single_prompt.py \"자동차 아니면 0 맞으면 1\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    print("="*80)
    print("단일 프롬프트 상세 테스트")
    print("="*80)
    print(f"\n테스트 프롬프트: '{prompt}'")
    print(f"프롬프트 길이: {len(prompt.encode('utf-8'))} bytes")
    
    # API 클라이언트 초기화
    client = OpenAI(api_key=API_KEY)
    
    # 데이터 로드
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples.csv')
    if not os.path.exists(data_path):
        print(f"ERROR: 데이터 파일을 찾을 수 없습니다: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"\n데이터: {len(df)}개 샘플 (자동차: {sum(df['label']==1)}, 비자동차: {sum(df['label']==0)})")
    
    # 테스트 실행
    predictions, outputs, errors = test_prompt_detailed(prompt, df, client)
    
    # 결과 분석
    results = analyze_results(df, predictions, outputs, prompt)
    
    # 결과 출력
    print("\n" + "="*80)
    print("테스트 결과")
    print("="*80)
    
    print(f"\n전체 정확도: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"  자동차 정확도: {results['car_accuracy']:.1%}")
    print(f"  비자동차 정확도: {results['non_car_accuracy']:.1%}")
    print(f"\n예상 점수: {results['score']:.4f}")
    print(f"  정확도 기여: {0.9 * results['accuracy']:.4f}")
    print(f"  길이 기여: {0.1 * np.sqrt(1 - (results['prompt_bytes']/3000)**2):.4f}")
    
    # 오분류 샘플 출력
    if results['misclassified']:
        print("\n" + "="*80)
        print(f"오분류 샘플 ({len(results['misclassified'])}개)")
        print("="*80)
        
        for i, miss in enumerate(results['misclassified'][:5], 1):
            print(f"\n{i}. 샘플 #{miss['index']}")
            print(f"   제목: {miss['title'][:50]}...")
            print(f"   실제: {miss['actual']}, 예측: {miss['predicted']}")
            print(f"   모델 출력: '{miss['output']}'")
        
        if len(results['misclassified']) > 5:
            print(f"\n... 외 {len(results['misclassified'])-5}개 더")
    else:
        print("\n완벽! 모든 샘플을 정확히 분류했습니다.")
    
    # 샘플 출력 예시
    print("\n" + "="*80)
    print("샘플 출력 예시 (처음 5개)")
    print("="*80)
    
    for i in range(min(5, len(outputs))):
        label = df.iloc[i]['label']
        pred = predictions[i]
        status = "✓" if pred == label else "✗"
        print(f"\n{i+1}. {status} 실제: {label}, 예측: {pred}")
        print(f"   제목: {df.iloc[i]['title'][:40]}...")
        print(f"   출력: '{outputs[i]}'")
    
    # 오류 정보
    if errors:
        print("\n" + "="*80)
        print(f"오류 발생 ({len(errors)}건)")
        print("="*80)
        for idx, error in errors:
            print(f"  샘플 #{idx}: {error}")

if __name__ == "__main__":
    main()