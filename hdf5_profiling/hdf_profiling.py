import h5py
import numpy as np
import pandas as pd

# HDF5 파일 경로
filename = './hdf5_file/i2048_b16.h5'

with h5py.File(filename, 'r') as f:
    # CUPTI_ACTIVITY_KIND_KERNEL 데이터셋 열기
    kernel_ds = f['CUPTI_ACTIVITY_KIND_KERNEL']
    
    # 전체 데이터를 numpy 배열로 로드
    kernel_data = kernel_ds[()]
    
    # numpy 배열을 pandas DataFrame으로 변환
    df = pd.DataFrame(kernel_data)
    
    # 커널 이름을 담을 새로운 컬럼 생성: 'demangledName'을 사용
    if 'demangledName' in df.columns:
        df['Kernel name'] = df['demangledName'].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )
    # 'demangledName'이 없는 경우, 'shortName' 사용 (필요시)
    elif 'shortName' in df.columns:
        df['Kernel name'] = df['shortName'].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )
    else:
        df['Kernel name'] = 'Unknown'
    
    # 실행 시간(duration) 계산: duration = end - start
    if 'start' in df.columns and 'end' in df.columns:
        df['duration'] = df['end'] - df['start']
    else:
        df['duration'] = None
    
    # 필요한 컬럼만 선택
    output_df = df[['Kernel name', 'start', 'end', 'duration']]
    
    # CSV 파일로 저장 (인덱스는 저장하지 않음)
    output_csv = 'kernel_data_clean.csv'
    output_df.to_csv(output_csv, index=False)
    print(f"CSV 파일이 '{output_csv}'로 저장되었습니다.")
