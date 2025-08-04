# download_model.py

import os
import ssl
from huggingface_hub import snapshot_download

# --- [SSL 오류 우회] 전역 인증서 검증 비활성화 ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- 우회 코드 끝 ---

# 다운로드할 모델 이름
model_name = "jhgan/ko-sroberta-multitask"

# 모델을 저장할 로컬 폴더 이름
local_model_path = "./ko-sroberta-multitask"

print(f"'{model_name}' 모델 다운로드를 시작합니다.")
print(f"저장 위치: {os.path.abspath(local_model_path)}")

# 폴더가 없으면 생성
os.makedirs(local_model_path, exist_ok=True)

# 모델 다운로드 실행
snapshot_download(
    repo_id=model_name,
    local_dir=local_model_path,
    local_dir_use_symlinks=False, # 윈도우 호환성을 위해 False로 설정
    resume_download=True
)

print("\n✅ 모델 다운로드가 성공적으로 완료되었습니다.")
print(f"이제 'vector_search_service.py'의 build_and_save_index()를 실행할 수 있습니다.")