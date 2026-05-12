# Pokémon Card AI Grader Setup Guide

## Requirements

Install:

- Python 3.10+
- Node.js 20+
- Git

---

# 1. Clone Repository

```bash
git clone https://github.com/BobC96/BobC96.git
```

---

# 2. Backend Setup

```bash
cd BobC96/pokemon-card-ai-grader/backend
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

## Windows

```bash
venv\\Scripts\\activate
```

## Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend:

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

---

# 3. Frontend Setup

Open another terminal:

```bash
cd BobC96/pokemon-card-ai-grader/frontend
```

Install dependencies:

```bash
npm install
```

Run frontend:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

# 4. Usage

1. Upload front scan
2. Upload back scan
3. Click Grade Card
4. Review grading output

---

# Recommended Scan Standard

- 1200 DPI PNG preferred
- Flatbed scanner recommended
- No scanner enhancement
- Clean scanner glass
- Card aligned vertically
