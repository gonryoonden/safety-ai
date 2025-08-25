import os
import json

LAWS_DIR = "laws"
ENV_FILE = ".env"

def extract_msts_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            law_data = data["LawSearch"]["law"]

            if isinstance(law_data, list):
                law = law_data[0]
            else:
                law = law_data

            return law["법령일련번호"], law.get("법령명한글", "(법령명 없음)")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"❌ MSTS 추출 실패: {filepath} → {type(e).__name__}")
            return None, None

def update_env_file(msts_list):
    msts_str = ",".join(map(str, msts_list))
    new_line = f"MSTS={msts_str}\n"
    
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(new_line)
        print(f"📄 .env 파일 생성됨 → {new_line.strip()}")
        return

    with open(ENV_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith("MSTS="):
            lines[i] = new_line
            updated = True
            break

    if not updated:
        lines.append(new_line)

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"🔄 .env 파일 업데이트 완료 → {new_line.strip()}")

def main():
    print("📦 laws 디렉토리에서 법령일련번호 추출 중...")
    results = []

    for filename in os.listdir(LAWS_DIR):
        if filename.endswith(".json"):
            full_path = os.path.join(LAWS_DIR, filename)
            mst_id, law_name = extract_msts_from_file(full_path)
            if mst_id:
                results.append((filename, mst_id))
                print(f"✅ {filename} → MSTS: {mst_id} | 법령명: {law_name}")
            else:
                print(f"⚠️  {filename} → MSTS 없음 또는 오류")

    msts_only = [str(m[1]) for m in results if m[1]]

    if msts_only:
        update_env_file(msts_only)
        print("\n📋 최종 추출 결과:")
        print("MSTS =", ",".join(msts_only))
    else:
        print("🚫 추출된 MSTS 없음")

if __name__ == "__main__":
    main()
