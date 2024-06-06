# Back-end part for our Ddobagi project

## 프로젝트 개요
이 프로젝트는 [또박이]입니다. 또박이는 한글 발음 교정 웹 서비스입니다.
해당 레포지토리는 이 서비스의 백엔드 부분입니다.

## 주요 기능
- 질문 게시 및 보기
- 답변 게시 및 보기
- 사용자 인증 및 권한 관리
- 질문에 대한 피드백 제공

## 기술 스택
- Python 3.11
- Django 5.0.4
- Django Rest Framework
- SqLite

## 프로젝트 설정 및 실행

### 1. 환경 설정
먼저 가상 환경을 설정합니다:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### 2. 프로젝트 clone
```bash
git clone https://github.com/CAU-CAPSTONE-2024-TEAM2/back-end.git
cd back-end
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. Sqlite 마이그레이션
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Superuser 생성 및 서버 실행
```bash
python manage.py createsuperuser
python manage.py runserver
```

## API 문서
API 문서: [Swagger](http://3.133.191.166:8000/swagger) [redoc](http://3.133.191.166:8000/redoc)
