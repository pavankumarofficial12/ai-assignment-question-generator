import os
import json
import re
import httpx
import boto3
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import botocore.exceptions
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# ========================= CONFIGURATION =========================
app = FastAPI(
    title="AI Assignment Question Generator",
    description="Generates personalized assignment questions using AWS Bedrock Llama 3",
    version="1.0.0"
)

# CORS Middleware (Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Configuration from .env
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise Exception("AWS credentials not found in environment variables")

# Initialize Bedrock Client
try:
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
except Exception as e:
    raise Exception(f"Failed to initialize Bedrock client: {e}")

# ========================= HELPER FUNCTIONS =========================
async def fetch_assignment_data(user_id: int, assignment_id: int):
    url = f"http://localhost:8080/ai-course-assignments/getDataByAsignmentIdAndUserId/{assignment_id}/{user_id}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Assignment data not found")
        return response.json()

def generate_prompt(data, main_topic_title, sub_topic):
    grade = data.get('grade', 'N/A')
    feedback = data.get('feedback', 'No feedback provided')
    score = data.get('score', 'N/A')
    week = data.get('weekNumber', 'Unknown')

    return f"""
You are an AI course instructor tasked with generating personalized assessment questions.

Student Performance:
- Grade: {grade}
- Score: {score}
- Feedback: {feedback}
- Week Number: {week}

Assignment Topic Details:
- Main Topic Title: {main_topic_title}
- Sub Topic: {sub_topic}

Based on the assignment topic and performance, generate ONLY 2 or 3 thoughtful, concept-driven, and student-level-appropriate questions that:

- Focus on the student's current performance level ({grade}, {feedback}, {score})
- For low performance: Generate basic foundational questions
- For moderate performance: Generate intermediate-level questions
- For high performance: Generate advanced, critical thinking questions

Return only the questions as a numbered list. Do not add explanations.
"""

def call_llama(prompt: str):
    try:
        body = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9
        }

        response = bedrock.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        response_body = json.loads(response['body'].read())
        return response_body.get("completion") or response_body.get("generation") or str(response_body)

    except botocore.exceptions.NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not configured")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate questions")

# ========================= REQUEST MODEL =========================
class AssignmentRequest(BaseModel):
    user_id: int
    assignment_id: int

# ========================= API ENDPOINT =========================
@app.get("/generate-assignment/{user_id}/{assignment_id}")
async def generate_assignment(user_id: int, assignment_id: int):
    try:
        data = await fetch_assignment_data(user_id, assignment_id)

        assignment_grade = data.get('AssignmentGrade', {})
        assignment = assignment_grade.get('aiCourseAssignment', {})
        plan_id = assignment.get('aiCoursePlan', {}).get('id')
        week_number = assignment.get('weekNumber')
        grade = assignment_grade.get('grade')
        score = assignment_grade.get('score')
        feedback = assignment_grade.get('feedback')

        # Extract topics
        topics = data.get('Topics', [])
        main_topic_titles = ', '.join([topic.get('mainTopicTitle', 'Unknown') for topic in topics]) if topics else 'Unknown'
        sub_topics = ', '.join([topic.get('subTopic', 'Unknown') for topic in topics]) if topics else 'Unknown'

        prompt = generate_prompt(assignment_grade, main_topic_titles, sub_topics)
        output = call_llama(prompt).strip()

        questions = re.findall(r'\d+\.\s+(.*)', output)
        if not questions:
            questions = [output.strip()]

        assignment_description = "This assignment covers key concepts based on student performance and topic."

        return {
            "user_id": user_id,
            "ai_course_plan_id": plan_id,
            "week_number": week_number,
            "grade": grade,
            "score": score,
            "feedback": feedback,
            "main_topic_title": main_topic_titles,
            "sub_topic": sub_topics,
            "Assignment Description": assignment_description,
            "Tasks": [q.strip() for q in questions]
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Assignment Question Generator"}

# ========================= RUN SERVER =========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
