import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# === Set up Gemini-compatible client ===
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Tool function ===
def get_career_roadmap(field: str) -> str:
    roadmaps = {
        "software engineering": "1. Learn Python or JavaScript\n2. Study data structures and algorithms\n3. Build real projects\n4. Contribute to open source\n5. Apply to internships/jobs",
        "data science": "1. Learn Python and statistics\n2. Study machine learning basics\n3. Practice on datasets (e.g. Kaggle)\n4. Build a portfolio\n5. Apply for junior roles",
        "medicine": "1. Take pre-med courses\n2. Prepare for MCAT\n3. Attend medical school\n4. Do clinical rotations\n5. Specialize",
        "marketing": "1. Learn digital marketing fundamentals\n2. Study consumer psychology\n3. Practice with social media campaigns\n4. Get certified in Google/Facebook Ads\n5. Build a portfolio of campaigns",
        "finance": "1. Learn financial accounting basics\n2. Study investment principles\n3. Get familiar with financial modeling\n4. Pursue CFA or similar certifications\n5. Apply for analyst positions"
    }
    return roadmaps.get(
        field.lower(),
        f"No roadmap found for '{field}'. Try asking about available fields like 'software engineering', 'data science', 'medicine', 'marketing', or 'finance'."
    )

# === Define Specialist Agents ===
skill_agent = Agent(
    name="SkillAgent",
    instructions="""You are a skill development specialist. When a user asks about learning skills, roadmaps, or how to develop expertise in a field, provide detailed guidance using the 'get_career_roadmap' tool. 
    
    Always use the tool to get the roadmap and then expand on it with additional helpful details, tips, and resources.""",
    tools=[function_tool(get_career_roadmap)]  # ✅ FIXED
)

job_agent = Agent(
    name="JobAgent",
    instructions="""You are a job market specialist. When users ask about job roles, positions, titles, or employment opportunities in a field, provide comprehensive information including:
    - Specific job titles and roles
    - Typical responsibilities
    - Salary ranges (if known)
    - Companies that hire for these roles
    - Career progression paths
    
    Focus on real-world, practical job market insights."""
)

career_exploration_agent = Agent(
    name="CareerExplorationAgent", 
    instructions="""You are a career exploration specialist. Help users discover career paths based on their interests, skills, and goals. Ask probing questions to understand their:
    - Interests and passions
    - Natural strengths and skills
    - Work environment preferences
    - Long-term career goals
    
    Provide personalized career suggestions and guide them toward next steps."""
)

# === Main Triage Agent with Handoffs ===
triage_agent = Agent(
    name="CareerMentorTriageAgent",
    instructions="""You are Career Mentor AI, a helpful career guidance system. Your role is to:

1. **Understand the user's query** and determine what type of help they need
2. **Route them to the right specialist**:
   - If they ask about SKILLS, LEARNING, ROADMAPS, or HOW TO DEVELOP expertise → hand off to SkillAgent
   - If they ask about JOBS, ROLES, POSITIONS, TITLES, or EMPLOYMENT → hand off to JobAgent  
   - If they want to EXPLORE careers, discuss INTERESTS, or need general career guidance → handle it yourself as the career exploration specialist

3. **Before any handoff**, briefly acknowledge their request and explain why you're connecting them to a specialist.

Start conversations by welcoming users and asking about their career interests or goals.""",
    handoffs=[skill_agent, job_agent]
)

# === Chat start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    await cl.Message(
        content=(
            "👋 Welcome to Career Mentor AI!\n\n"
            "I'm here to help you with your career journey. I can assist you with:\n"
            "• **Career Exploration** - Discover paths based on your interests\n"
            "• **Skill Development** - Get detailed learning roadmaps\n"
            "• **Job Market Info** - Learn about roles and opportunities\n\n"
            "What would you like to explore today?"
        )
    ).send()

# === Message handling ===
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Use the triage agent - it will automatically handle handoffs
        result = Runner.run_streamed(
            triage_agent, history, run_config=cast(RunConfig, config)
        )
        
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                await msg.stream_token(event.data.delta)

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")
        print(f"Error: {e}")
