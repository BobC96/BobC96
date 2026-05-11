from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pokemon Card AI Grader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Pokemon Card AI Grader API Running"
    }

@app.post("/grade")
async def grade_card(
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...)
):
    return {
        "estimated_grade": 9,
        "centering_score": 9.5,
        "corners_score": 9,
        "edges_score": 8.5,
        "surface_score": 9,
        "confidence": 0.72,
        "message": "Initial MVP placeholder grading response"
    }
